#include <deque>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "asynccapture.hpp"
#include "parser.hpp"
#include "particle.hpp"

#include "cpuproc.hpp"
#include "gpuproc.hpp"
#include "io.hpp"
#include "util.hpp"

#include "tracy/Tracy.hpp"

int main(int argc, char *argv[]) {

  // find and check parameters
  std::filesystem::path path(argv[1]);

  int particle_image_scale = 1;
  bool export_images = false;

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
  auto cap = AsyncVideoCapture(path, cv::CAP_FFMPEG);

  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  int frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);

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
  cv::Mat cpu_frame, cpu_mask;
  cap.read(cpu_frame);
  cv::cvtColor(cpu_frame, cpu_frame, cv::COLOR_BGR2GRAY);

  std::cout << "Processsing " << path << std::endl;

  double um_per_px;
  if (mask_capillary(cpu_frame, cpu_mask, um_per_px)) {
    return 1;
  }

  std::cout << "\tframes = " << frame_count << std::endl;
  std::cout << "\tsize = " << width << " x " << height << std::endl;
  std::cout << "\tµm per px = " << um_per_px << std::endl;

  // setup arrays
  cv::cuda::GpuMat frame(cpu_frame);
  cv::cuda::GpuMat mask(cpu_mask);
  cv::cuda::GpuMat acc_mean;
  cv::cuda::GpuMat acc_var =
      cv::cuda::GpuMat(cpu_frame.rows, cpu_frame.cols, CV_32F);
  auto stream = cv::cuda::Stream();

  // cv::Mat frame(cpu_frame);
  // cv::Mat mask(cpu_mask);
  // cv::Mat acc_mean;
  // cv::Mat acc_var = cv::Mat(frame.rows, frame.cols, CV_32F);

  // init the accumulated mean and variance
  frame.convertTo(acc_mean, CV_32F);
  acc_var.setTo(0.f);

  // begin by reading the required number of frames to predict the background
  init_background(cap, acc_mean, acc_var, background_frames);

  if (draw_frames) {
    cv::namedWindow("frame", cv::WINDOW_NORMAL);
    // cv::namedWindow("diff", cv::WINDOW_NORMAL);
  };

  // reset the video
  int frame_pos = 0;
  cap.set(cv::CAP_PROP_POS_FRAMES, frame_pos);
  cap.invalidate();
  auto start_time = std::chrono::system_clock::now();

  // init the particle vars
  std::deque<std::vector<Particle>> particles;
  int particle_id = 0;
  int particle_count = 0;

  cap.read(cpu_frame);
  frame.upload(cpu_frame, stream);

  while (frame_pos++ < frame_count) {
    {
      ZoneScopedN("read frame");
      stream.waitForCompletion();

      cap.read(cpu_frame);
      frame.upload(cpu_frame);
      // read in a new frame
      if (cpu_frame.empty()) {
        break;
      }
      cv::cuda::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    }

    // update the background accumulated mean and variance,
    // if on an unread frame
    if (frame_pos > background_frames) {
      ZoneScopedN("update background");
      update_background(frame, acc_mean, acc_var, frame_pos);
    }

    std::vector<Particle> new_particles;
    {
      ZoneScopedN("find");
      find_particles(frame, acc_mean, acc_var, zscore, mask, new_particles,
                     frame_pos, particle_id);
    }
    {
      std::cout << "particle pre filter: " << new_particles.size() << std::endl;
      ZoneScopedN("filter");
      // filter particle based on parameters
      filter_particles(new_particles, particle_filter_args);
      std::cout << "particle mid filter: " << new_particles.size() << std::endl;
      // filter based on last n frames
      for (auto it = particles.begin(); it != particles.end(); ++it) {
        filter_existing_particles(
            *it, new_particles,
            [](const Particle &a, const Particle &b) {
              return a.centerWeightedIntensity() <= b.centerWeightedIntensity();
            },
            particle_distance);
      }
      std::cout << "particle post filter: " << new_particles.size() << std::endl;
      particle_count += new_particles.size();
      particles.push_back(new_particles);
    }

    // create a color image and draw the contuors
    if (draw_frames) {
      auto color = cv::Scalar(0, 0, 255);
      int decay = 255 / particle_frames;
      std::vector<std::vector<cv::Point>> contours;
      for (auto it = particles.rbegin(); it != particles.rend(); ++it) {
        contours.resize(it->size());
        std::transform(std::execution::par, it->begin(), it->end(),
                       contours.begin(),
                       [](const Particle &p) { return p.contour(); });
        cv::drawContours(cpu_frame, contours, -1, color, 1.0, 8);
        color[2] -= decay;
      }
      // get the filtered contours
      cv::imshow("frame", cpu_frame);
      // cpu_diff.convertTo(cpu_diff, -1, 1.0 / 255.0, 0.5);
      // cv::imshow("diff", cpu_diff);

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
        if (write_particle_images(output_particles, image_dir)) {
          return 1;
        }
      }
      particles.pop_front();
    }

    // update progress
    if (frame_pos % 100 == 0) {
      double fps;
      auto remaining =
          get_remaining_time(start_time, frame_pos, frame_count, fps);

      std::cout << "\t...processing :: frame " << frame_pos << "/"
                << frame_count << " @ ";
      std::cout << std::setw(3) << static_cast<int>(fps) << " FPS, ";
      std::cout << particle_count << " particles, ";
      std::cout << std::format("{:%T}", remaining) << " remaining.\r"
                << std::flush;
    }

  } // while

  // export any remaining particles
  for (auto it = particles.begin(); it != particles.end(); ++it) {
    write_particle_data(*it, results_output);
    if (export_images) {
      if (write_particle_images(*it, image_dir)) {
        return 1;
      }
    }
  }

  cv::Mat acc_out(acc_mean);
  acc_out.convertTo(acc_out, CV_8U);
  cv::imwrite(proc_dir / "background_mean.png", acc_out);
  cv::Mat acc_var_out(acc_var);
  acc_var_out.convertTo(acc_var_out, CV_8U);
  cv::imwrite(proc_dir / "background_var.png", acc_var_out);

  auto total_duration = std::chrono::duration<double>(
      std::chrono::system_clock::now() - start_time);

  std::cout << std::endl
            << "Finished in " << std::format("{:%T}", total_duration)
            << std::endl;

  return 0;
}
