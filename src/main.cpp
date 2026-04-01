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

class Parser {
private:
  std::vector<std::string> args;

public:
  Parser(int argc, char *argv[]) {
    args = std::vector<std::string>(argv, argv + argc);
  }

  template <typename T>
  T read(const std::string &name, const T &default_value,
         bool required = false) {
    T value = default_value;
    for (auto it = args.begin(); it != args.end(); ++it) {
      if ((*it).substr(2) == name) {
        // shortcut for flags
        if (std::is_same<T, bool>::value) {
          return true;
        }
        ++it;
        std::istringstream iss(*it);
        if (!(iss >> value)) {
          throw std::invalid_argument("unable to read '" + *it +
                                      "' into arg '" + name + "'");
        };
        return value;
      }
    }
    if (required) {
      throw std::invalid_argument("missing argument '" + name + "'");
    }
    return value;
  }

  // template <typename T>
  // void read(const std::string &name, T &value, bool required = true) {
  //   for (auto it = args.begin(); it != args.end(); ++it) {
  //     if ((*it).substr(2) == name) {
  //       // shortcut for flags
  //       if (std::is_same<T, bool>::value) {
  //         value = true;
  //         return;
  //       }
  //       ++it;
  //       std::istringstream iss(*it);
  //       if (!iss >> value) {
  //         throw std::invalid_argument("unable to read '" + *it +
  //                                     "' into arg '" + name + "'");
  //       };
  //       return;
  //     }
  //   }
  //   if (required) {
  //     throw std::invalid_argument("missing argument '" + name + "'");
  //   }
  // }
};

std::pair<cv::Mat, cv::Mat> measure_background(cv::VideoCapture &cap,
                                               double time) {

  int fps = cap.get(cv::CAP_PROP_FPS);
  int max_frames = static_cast<int>(fps * time);

  std::vector<cv::Mat> frames;
  frames.reserve(max_frames);

  cv::Mat frame;
  int current_frame = 0;
  while (current_frame++ < max_frames) {
    cap.read(frame);
    if (frame.empty()) {
      std::cerr << "insufficient background frames, exiting early" << std::endl;
      break;
    }
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    frame.convertTo(frame, CV_32F, 1.0 / 255.0);
    frames.push_back(frame);
  }
  int n_frames = frames.size();

  cv::Mat mean = std::transform_reduce(
      frames.begin(), frames.end(),
      cv::Mat::zeros(frame.size.dims(), frame.size, CV_32F), std::plus{},
      [&](cv::Mat &val) { return val / n_frames; });

  cv::Mat var = std::transform_reduce(
      frames.begin(), frames.end(),
      cv::Mat::zeros(frame.size.dims(), frame.size, CV_32F), std::plus{},
      [&](cv::Mat &val) {
        cv::Mat x;
        cv::pow(val - mean, 2.0, x);
        return x / (n_frames - 1);
      });
  cv::sqrt(var, var);

  mean.convertTo(mean, CV_8U, 255.0);
  var.convertTo(var, CV_8U, 255.0);

  return {mean, var};
}

bool find_camera_roi(const cv::Mat &mean, cv::Vec3f &roi) {
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(mean, circles, cv::HOUGH_GRADIENT, 1.0,
                   static_cast<int>(mean.rows / 2), 50, 5, mean.rows / 4);

  if (circles.size() == 0) {
    std::cerr << "could not detect frame ROI, exiting" << std::endl;
    return true;
  }
  std::cout << "ROI detected at " << circles[0][0] << " x " << circles[0][1]
            << " with radius " << circles[0][2] << std::endl;

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
                             // /
                             //                       it_new->area();
                             auto norm_old = old.centerWeightedIntensity();
                             // / old.area();
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

void update_progress(
    std::chrono::time_point<std::chrono::system_clock> start_time, int frame,
    int frame_count) {
  auto frame_time = std::chrono::system_clock::now();
  auto duration = std::chrono::duration<double>(frame_time - start_time);
  double fps = static_cast<double>(frame) / duration.count();
  std::chrono::duration remaining =
      duration *
          (static_cast<double>(frame_count) / static_cast<double>(frame)) -
      duration;

  std::cout << "Processing frame " << frame << "/" << frame_count << " @ ";
  std::cout << std::setw(5) << std::setprecision(5) << fps << " FPS. ";
  std::cout << std::format("{:%T}", remaining) << " \r" << std::flush;
}

cv::Mat richardson_lucy(const cv::Mat &input, const cv::Mat &psf,
                        int iterations = 3) {

  cv::Mat est = cv::Mat(2, input.size, CV_32F, 0.5);
  cv::Mat psf_hat;
  cv::flip(psf, psf_hat, -1);

  cv::Mat est_conv, relative_blur, error_est;
  for (int i = 0; i < iterations; ++i) {
    cv::filter2D(est, est_conv, -1, psf);
    relative_blur = input.mul(1.0 / est_conv);
    cv::filter2D(relative_blur, error_est, -1, psf_hat);
    est = est.mul(error_est);
  }
  return est;
}

cv::Mat gaussian_kernel(int rows, int cols, float sigma) {
  cv::Mat kernel = cv::Mat::zeros(rows, cols, CV_32F);
  kernel.at<float>(rows / 2, cols / 2) = 1.f;
  cv::GaussianBlur(kernel, kernel, cv::Size(rows - 1, cols - 1), sigma);
  return kernel;
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cerr << "missing argument: lpc {VIDEO_FILE}" << std::endl;
    return 1;
  }
  // find and check parameters
  std::filesystem::path path(argv[1]);

  double roi_size_um = 750.0;
  int particle_image_scale = 1;
  bool export_images = true;

  auto parser = Parser(argc, argv);
  int background_frames = parser.read("background-frames", 1000);
  int particle_frames = parser.read("particle-frames", 10);
  double particle_distance = parser.read("particle-distance", 10.0);
  double zscore = parser.read("zscore", 3.0);
  bool draw_frames = parser.read("draw", false);

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

  cv::Vec3f roi;
  if (find_camera_roi(frame, roi)) {
    return 1;
  } else {
    cv::Mat roi_frame;
    cv::cvtColor(frame, roi_frame, cv::COLOR_GRAY2BGR);
    cv::circle(roi_frame, cv::Point(roi[0], roi[1]), roi[2],
               cv::Scalar(0, 0, 255), 1);
    cv::imwrite(proc_dir / "roi.png", roi_frame);
  }
  // calculate the pixel size
  double um_per_px = roi_size_um / (2.0 * roi[2]);
  std::cout << "Pixel size = " << std::setprecision(4) << um_per_px << " µm"
            << std::endl;

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
  std::cout << "Processing " << nframes << " frames..." << std::endl;

  int frames = 1;
  int particle_id = 0;
  while (true) {

    // read in a new frame
    cap.read(frame);
    if (frame.empty()) {
      break;
    }
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    // update every second or so
    if (frames % 100 == 0) {
      update_progress(start_time, frames, nframes);
    }
    frames++;

    // update the accumulated mean and variance
    cv::accumulateWeighted(frame, acc_mean, 1.0 / static_cast<double>(frames));
    cv::Mat frame_var;
    frame.convertTo(frame_var, CV_32F);
    cv::pow(frame_var - acc_mean, 2.0, frame_var);
    cv::accumulateWeighted(frame_var, acc_var,
                           1.0 / static_cast<double>(frames));

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
    // diff = cv::abs(diff);

    // mask differences below x std deviations
    cv::Mat thresh = cv::Mat::zeros(2, diff.size, CV_8U);
    // if (mode == 0) {
    //   cv::bitwise_and(cv::abs(diff) > zscore * std, mask, thresh);
    // } else if (mode < 0) {
    diff *= -1;
    cv::bitwise_and(diff > zscore * std, mask, thresh);
    // } else {
    //   cv::bitwise_and(diff > 0.5 * zscore * std, mask, thresh);
    // }
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
    filter_particles(new_particles, {
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
                                    });
    // filter based on last n frames
    filter_existing_particles(particles, new_particles, particle_distance);
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
  }

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
  std::cout << "Finished" << std::endl;

  return 0;
}
