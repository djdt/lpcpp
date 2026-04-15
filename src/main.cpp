#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "parser.hpp"
#include "particle.hpp"

#include "cpuproc.hpp"
#include "io.hpp"
#include "util.hpp"

int main(int argc, char *argv[]) {

  // find and check parameters
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
  double unsharp =
      parser.read("unsharp", 1.0, "apply an unsharp mask at this alpha");
  bool draw_frames = parser.read("draw", false, "show video and detections");
  std::string output_path = parser.read<std::string>(
      "output", std::string(), "output directory for processed data");
  bool export_images =
      parser.read("export-images", false, "export images of particles");
  std::string config_path = parser.read<std::string>(
      "config", std::string(),
      "path to filter config, with lines: 'VALUE MIN MAX'\n"
      "\tvalid values are: 'area', 'aspect', circularity', 'convexity', "
      "'radius'");

  filter_args particle_filter_args;

  if (!config_path.empty()) {
    read_filter_config(config_path, particle_filter_args);
  }

  if (argc < 2 || !parser.success()) {
    std::cerr << "Usage: lpcpp FILE [options]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << parser;
    return 1;
  }

  std::filesystem::path path(argv[1]);

  // create capture and read some props
  auto cap = cv::VideoCapture(path.string(), cv::CAP_FFMPEG);
  if (!cap.set(cv::CAP_PROP_CONVERT_RGB, 0)) {
    std::cerr << "cannot read as greyscale" << std::endl;
    return 1;
  }

  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  int frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);

  // create output directory
  std::filesystem::path output_dir;
  if (output_path.empty())
    output_dir = path.parent_path() / "processed";
  else {
    output_dir = std::filesystem::path(output_path);
    if (std::filesystem::exists(output_path) &&
        !std::filesystem::is_directory(output_path)) {
      std::cerr << "output path" << output_dir << "is not a directory"
                << std::endl;
      return 1;
    }
  }

  std::filesystem::create_directory(output_dir);
  auto image_dir = output_dir / "particles";
  if (export_images) {
    std::filesystem::create_directory(image_dir);
  }
  std::ofstream results_output(output_dir / "particles.csv", std::ios::out);
  write_particle_header(results_output);

  // load a frame and find the ROI
  cv::UMat frame, mask;
  cap.read(frame);

  std::cout << "Processsing " << path << std::endl;

  std::cout << "\tframes = " << frame_count << std::endl;
  std::cout << "\tsize = " << width << " x " << height << std::endl;

  double um_per_px;
  if (mask_capillary(frame, mask, um_per_px)) {
    std::cerr << "\tcould not detect capillary" << std::endl;
    return 1;
  }
  std::cout << "\tµm per px = " << um_per_px << std::endl;

  // setup arrays
  cv::UMat acc_mean;
  cv::UMat acc_var = cv::UMat(frame.rows, frame.cols, CV_32F);

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
  auto start_time = std::chrono::system_clock::now();

  // init the particle vars
  std::deque<std::vector<Particle>> particles;
  int particle_id = 0;
  int particle_count = 0;

  while (frame_pos++ < frame_count) {
    cap.read(frame);

    // read in a new frame
    if (frame.empty()) {
      break;
    }

    // update the background accumulated mean and variance,
    // if on an unread frame
    if (frame_pos > background_frames) {
      update_background(frame, acc_mean, acc_var, frame_pos);
    }

    std::vector<Particle> new_particles;
    find_particles(frame, acc_mean, acc_var, zscore, mask, unsharp,
                   new_particles, frame_pos, particle_id);
    // filter particle based on parameters
    filter_particles(new_particles, particle_filter_args);
    // filter based on last n frames
    for (auto it = particles.begin(); it != particles.end(); ++it) {
      filter_existing_particles(
          *it, new_particles,
          [](const Particle &a, const Particle &b) {
            return a.centerWeightedIntensity() > b.centerWeightedIntensity();
          },
          particle_distance);
    }
    particles.push_back(new_particles);

    // create a color image and draw the contuors
    if (draw_frames) {
      cv::UMat rgb_frame;
      draw_particles_on_frame(frame, rgb_frame, particles.rbegin(),
                              particles.rend(), particle_frames);
      cv::imshow("frame", frame);

      int key = cv::waitKey(50);
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
      particle_count += particles.size();
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

  cv::UMat acc_out(acc_mean);
  acc_out.convertTo(acc_out, CV_8U);
  cv::imwrite((output_dir / "background_mean.png").string(), acc_out);
  cv::UMat acc_var_out(acc_var);
  acc_var_out.convertTo(acc_var_out, CV_8U);
  cv::imwrite((output_dir / "background_var.png").string(), acc_var_out);

  auto total_duration = std::chrono::duration<double>(
      std::chrono::system_clock::now() - start_time);

  std::cout << std::endl
            << "Finished in " << std::format("{:%T}", total_duration)
            << std::endl;

  return 0;
}
