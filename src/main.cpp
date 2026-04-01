#include <deque>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <string>

#include "parser.hpp"
#include "particle.hpp"

template <typename Iter, typename IdxIter>
auto remove_indices(Iter begin, Iter end, IdxIter indices_begin,
                    IdxIter indices_end) -> Iter {

  while (indices_end != indices_begin) {
    --indices_end;
    auto pos = begin + *indices_end;
    std::move(std::next(pos), end--, pos);
  }
  return end;
}

bool find_camera_roi(const cv::Mat &mean, cv::Vec3f &roi) {
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(mean, circles, cv::HOUGH_GRADIENT, 1.0,
                   static_cast<int>(mean.rows / 2), 50, 5, mean.rows / 4);

  if (circles.size() == 0) {
    return true;
  }
  roi = circles[0];
  return false;
}

void write_particle_header(std::ofstream &ofs) {
  ofs << "id,frame,frame_count,area,aspect,circularity,convexity,intensity,"
         "radius,radius_at_median,x,y"
      << std::endl;
}
void write_particle_data(const std::vector<Particle> &particles,
                         std::ofstream &ofs) {
  for (auto it = particles.begin(); it != particles.end(); ++it) {
    ofs << it->id() << "," << it->frame_number() << "," << it->frame_count()
        << ",";
    ofs << it->area() << "," << it->aspect() << "," << it->circularity() << ",";
    ofs << it->convexity() << "," << it->intensity() << "," << it->radius()
        << "," << it->radiusAtQuantile(0.5) << ",";
    ofs << it->center().x << "," << it->center().y << std::endl;
  }
}
bool export_particle_images(const std::vector<Particle> &particles,
                            const std::filesystem::path &output_dir) {
  auto color = cv::Scalar(0, 0, 255);
  for (auto it = particles.begin(); it != particles.end(); ++it) {
    auto out = output_dir / std::to_string(it->id()).append(".png");
    cv::Mat image;
    it->image().convertTo(image, CV_8U);
    cv::Mat rgb = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    cv::insertChannel(image, rgb, 1);

    cv::Mat fill = cv::Mat::zeros(image.rows, image.cols, CV_8U);
    // cv::polylines(rgb, it->imageContour(), -1, color, 1.0, 8);
    cv::fillPoly(fill, it->imageContour(), 255);
    cv::insertChannel(fill, rgb, 2);
    if (not cv::imwrite(out, rgb)) {
      std::cerr << "failed to save " << out << std::endl;
      return true;
    }
  }
  return false;
}

void filter_existing_particles(std::deque<std::vector<Particle>> &particles,
                               std::vector<Particle> &new_particles,
                               double edge_distance = 20.0) {
  /* Lookback through existing particles to find any that overlap with new
   * ones. If an overlap is found the particle with the greatest intensity is
   * kept. */
  for (auto it_old = particles.begin(); it_old != particles.end(); ++it_old) {
    std::vector<size_t> remove_new_at;
    it_old->erase(
        std::remove_if(std::execution::seq, it_old->begin(), it_old->end(),
                       [&](Particle &old) {
                         for (auto it_new = new_particles.begin();
                              it_new != new_particles.end(); ++it_new) {
                           if (it_new->is_close(old, edge_distance)) {
                             auto norm_new = it_new->centerWeightedIntensity();
                             auto norm_old = old.centerWeightedIntensity();
                             if (norm_new > norm_old) {
                               it_new->addFrame();
                               return true; // old is removed
                             } else {
                               // remove new
                               size_t idx =
                                   std::distance(new_particles.begin(), it_new);
                               if (remove_new_at.size() == 0 or
                                   remove_new_at.back() != idx) {
                                 remove_new_at.push_back(idx);
                               }
                               old.addFrame();
                               return false;
                             }
                           }
                         }
                         return false;
                       }),
        it_old->end());

    // sort and remove non-unqiue indicies
    std::sort(remove_new_at.begin(), remove_new_at.end());
    auto last = std::unique(remove_new_at.begin(), remove_new_at.end());
    remove_new_at.erase(last, remove_new_at.end());

    new_particles.erase(
        remove_indices(new_particles.begin(), new_particles.end(),
                       remove_new_at.begin(), remove_new_at.end()),
        new_particles.end());
  }
}

std::chrono::duration<double> get_remaining_time(
    std::chrono::time_point<std::chrono::system_clock> start_time, int n,
    int count, double &fps) {
  auto frame_time = std::chrono::system_clock::now();
  auto duration = std::chrono::duration<double>(frame_time - start_time);
  std::chrono::duration remaining =
      duration * (static_cast<double>(count) / static_cast<double>(n)) -
      duration;
  fps = static_cast<double>(n) / duration.count();
  return remaining;
}

void read_filter_config(std::string path, filter_args &args) {
  std::fstream ifs(path);
  std::string line;
  while (!ifs.eof()) {
    ifs >> line;
    if (line == "area")
      ifs >> args.min_area >> args.max_area;
    else if (line == "aspect")
      ifs >> args.min_aspect >> args.max_aspect;
    else if (line == "circularity")
      ifs >> args.min_circularity >> args.max_circularity;
    else if (line == "convexity")
      ifs >> args.min_convexity >> args.max_convexity;
    else if (line == "radius")
      ifs >> args.min_radius >> args.max_radius;
  }
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cerr << "missing argument: lpcpp {VIDEO_FILE}" << std::endl;
    return 1;
  }
  // find and check parameters
  std::filesystem::path path(argv[1]);

  double roi_size_um = 750.0;
  int particle_image_scale = 1;
  bool export_images = true;

  auto parser = ArgumentParser(argc, argv);
  int background_frames = parser.read(
      "background-frames", 1000,
      "number of background frames used to determine initial mean and std");
  int particle_frames =
      parser.read("particle-frames", 10,
                  "number of frames to track particles after last detection");
  double particle_distance = parser.read("particle-distance", 10.0,
                                         "minimum distance between particles");
  double zscore = parser.read(
      "zscore", 3.0, "number of std above the background mean to threshold");
  bool draw_frames = parser.read("draw", false, "show video and detections");
  std::string config_path = parser.read<std::string>(
      "config", std::string(),
      "path to filter config, with lines: 'VALUE MIN MAX'\n"
      "\tvalid values are: 'area', 'aspect', circularity', 'convexity', "
      "'radius'");

  filter_args particle_filter_args{
      .min_area = 5.0,
      .max_area = 9999.0,
      .min_aspect = 0.7,
      .max_aspect = 1.0,
      .min_circularity = 0.8,
      .max_circularity = 1.0,
      .min_convexity = 0.9,
      .max_convexity = 1.0,
      .min_radius = 1.0,
      .max_radius = 200.0,
  };
  if (!config_path.empty()) {
    read_filter_config(config_path, particle_filter_args);
  }

  if (!parser.success()) {
    std::cerr << "Usage: lpcpp FILE [options]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << parser;
    return 1;
  }

  // create capture and read some props
  auto cap = cv::VideoCapture(path, cv::CAP_FFMPEG);
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  int nframes = cap.get(cv::CAP_PROP_FRAME_COUNT);

  // create output directory
  auto proc_dir = path.parent_path() / "processed";
  std::filesystem::create_directory(proc_dir);
  auto image_dir = proc_dir / "particles";
  if (export_images) {
    std::filesystem::create_directory(image_dir);
  }
  std::ofstream results_output(proc_dir / "particles.csv", std::ios::out);
  write_particle_header(results_output);

  // load a frame and find the ROI
  cv::Mat frame;
  cap.read(frame);
  cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

  std::cout << "Processsing " << path << std::endl;
  std::cout << "\tframes = " << nframes << std::endl;
  std::cout << "\tsize = " << width << " x " << height << std::endl;

  cv::Vec3f roi;
  if (find_camera_roi(frame, roi)) {
    std::cerr << "\tcould not detect frame ROI, exiting" << std::endl;
    return 1;
  } else {
    std::cout << "\tROI detected at " << roi[0] << " x " << roi[1]
              << " with radius " << roi[2] << std::endl;
    //
    // cv::Mat roi_frame;
    // cv::cvtColor(frame, roi_frame, cv::COLOR_GRAY2BGR);
    // cv::circle(roi_frame, cv::Point(roi[0], roi[1]), roi[2],
    //            cv::Scalar(0, 0, 255), 1);
    // cv::imwrite(proc_dir / "roi.png", roi_frame);
  }

  // calculate the pixel size
  double um_per_px = roi_size_um / (2.0 * roi[2]);
  std::cout << "\tpixel size = " << std::setprecision(4) << um_per_px << " µm";
  std::cout << std::endl << std::endl;

  cv::Mat mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
  cv::circle(mask, cv::Point(roi[0], roi[1]), roi[2] * 0.9, 255, -1);

  // init the accumulated mean and variance
  cv::Mat acc_mean;
  frame.convertTo(acc_mean, CV_32F);
  cv::Mat acc_var = cv::Mat::zeros(2, frame.size, CV_32F);

  if (draw_frames) {
    cv::namedWindow("frame", cv::WINDOW_NORMAL);
    cv::namedWindow("diff", cv::WINDOW_NORMAL);
  };

  std::deque<std::vector<Particle>> particles;

  auto start_time = std::chrono::system_clock::now();
  int frames = 1;
  int particle_id = 0;
  int particle_count = 0;
  while (true) {

    // read in a new frame
    cap.read(frame);
    if (frame.empty()) {
      break;
    }
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

    // update the accumulated mean and variance
    cv::accumulateWeighted(frame, acc_mean, 1.0 / static_cast<double>(frames));
    cv::Mat frame_var;
    frame.convertTo(frame_var, CV_32F);
    cv::pow(frame_var - acc_mean, 2.0, frame_var);
    cv::accumulateWeighted(frame_var, acc_var,
                           1.0 / static_cast<double>(frames));

    // update progress every second or so
    if (frames % 100 == 0) {
      double fps;
      auto remaining = get_remaining_time(start_time, frames, nframes, fps);

      std::cout << "\t...frame " << frames << "/" << nframes << " @ ";
      std::cout << std::setw(3) << static_cast<int>(fps) << " FPS, ";
      std::cout << particle_count << " particles, ";
      std::cout << std::format("{:%T}", remaining) << " remaining.\r"
                << std::flush;
    }
    frames++;

    if (frames < background_frames) {
      continue;
    }

    // get std
    cv::Mat std;
    cv::sqrt(acc_var, std);

    // calculate the difference between frame and mean
    cv::Mat diff;
    frame.convertTo(diff, CV_32F);
    diff -= acc_mean;
    cv::medianBlur(diff, diff, 3);

    // mask differences below x std deviations
    cv::Mat thresh = cv::Mat::zeros(2, diff.size, CV_8U);
    diff *= -1;
    cv::bitwise_and(diff > zscore * std, mask, thresh);
    // remove contour bound
    cv::erode(thresh, thresh, cv::Mat());

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    std::vector<Particle> new_particles;
    new_particles.reserve(contours.size());
    std::transform(contours.begin(), contours.end(),
                   std::back_inserter(new_particles),
                   [&](const std::vector<cv::Point> &contour) {
                     return Particle(contour, diff, frames, particle_id++,
                                     particle_image_scale);
                   });

    // filter particle based on parameters
    filter_particles(new_particles, particle_filter_args);
    // filter based on last n frames
    filter_existing_particles(particles, new_particles, particle_distance);
    particle_count += new_particles.size();
    particles.push_back(new_particles);

    // create a color image and draw the contuors
    if (draw_frames) {
      cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
      auto color = cv::Scalar(0, 0, 255);
      int decay = 255 / particle_frames;
      for (auto it = particles.rbegin(); it != particles.rend(); ++it) {
        contours.resize(it->size());
        std::transform(std::execution::par, it->begin(), it->end(),
                       contours.begin(),
                       [](const Particle &p) { return p.contour(); });
        cv::drawContours(frame, contours, -1, color, 1.0, 8);
        color[2] -= decay;
      }
      // get the filtered contours
      cv::imshow("frame", frame);
      diff.convertTo(diff, -1, 1.0 / 255.0, 0.5);
      cv::imshow("diff", diff);

      int key = cv::waitKey(5000);
      if (key == 'q') {
        break;
      }
    }

    // output the particles
    if (particles.size() > particle_frames) {
      auto output_particles = particles.front();

      write_particle_data(output_particles, results_output);
      if (export_images) {
        if (export_particle_images(output_particles, image_dir)) {
          return 1;
        }
      }
      particles.pop_front();
    }

  } // while

  for (auto it = particles.begin(); it != particles.end(); ++it) {
    write_particle_data(*it, results_output);
    if (export_images) {
      if (export_particle_images(*it, image_dir)) {
        return 1;
      }
    }
  }

  cv::Mat acc_out;
  acc_mean.convertTo(acc_out, CV_8U);
  cv::imwrite(proc_dir / "background_mean.png", acc_mean);
  cv::Mat acc_var_out;
  acc_var.convertTo(acc_var_out, CV_8U);
  cv::imwrite(proc_dir / "background_var.png", acc_var);
  std::cout << "Finished" << path << std::endl;

  return 0;
}
