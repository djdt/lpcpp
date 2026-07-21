#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "CLI11.hpp"

#include "contours.hpp"
#include "cpuproc.hpp"
#include "io.hpp"
#include "particle.hpp"
#include "util.hpp"

#ifndef CMAKE_PROJECT_VERSION
#define CMAKE_PROJECT_VERSION "0.0.0"
#endif

int main(int argc, char *argv[]) {

  CLI::App app;
  app.option_defaults()->always_capture_default();
  app.set_help_flag("");
  app.set_help_all_flag("-h,--help");

  std::string inname, outname;

  int background_frames = 1000;
  int particle_frames = 50;

  double particle_distance = 5.0;
  double zscore = 3.0;
  double unsharp_alpha = 1.0;

  std::array<float, 3> capillary = {0.f, 0.f, 0.f};

  bool draw = false;
  bool export_images = false;
  bool export_hdf5 = false;
  bool create_config = false;

  ParticleMetric selection_metric = PM_CENTER_WEIGHTED_DARK;
  std::map<std::string, ParticleMetric> metric_map = {
      {"center", PM_CENTER_WEIGHTED_DARK},
      {"centerWhite", PM_CENTER_WEIGHTED_LIGHT},
      {"centerAbsolute", PM_CENTER_WEIGHTED_ABS},
      {"intensity", PM_AVERAGE_DARK},
      {"intensityWhite", PM_AVERAGE_LIGHT},
      {"intensityAbsolute", PM_AVERAGE_ABS},
      {"sharpness", PM_SHARPNESS}};

  filter_args contour_filter_args;

  auto file_opt =
      app.add_option("file", inname, "path to the captured OIM video")
          ->required()
          ->check(CLI::ExistingFile, "FileExists")
          ->configurable(false);
  app.add_option("--output,-o", outname,
                 "specify the output directory, defaults to 'processed'")
      ->check(CLI::NonexistentPath | CLI::ExistingDirectory)
      ->configurable(false);

  app.add_option("--selection-metric,-m", selection_metric,
                 "method of selecting the particle frame for processing. "
                 "Currently center weighted intensity, average intensity and "
                 "sharpness are implemented")
      ->transform(CLI::CheckedTransformer(metric_map, CLI::ignore_case));
  app.add_option(
         "--background", background_frames,
         "number of background frames used to determine initial mean and std")
      ->check(CLI::PositiveNumber);
  app.add_option("--track", particle_frames,
                 "number of frames to track particles after last detection")
      ->check(CLI::PositiveNumber);
  app.add_option("--distance", particle_distance,
                 "minimum distance between particles")
      ->check(CLI::PositiveNumber);
  app.add_option("--zscore", zscore,
                 "number of std above the background mean to threshold")
      ->check(CLI::PositiveNumber);
  app.add_option("--unsharp", unsharp_alpha, "alpha value of the unsharp mask")
      ->check(CLI::NonNegativeNumber);
  app.add_option("--capillary", capillary,
                 "capillary position and radius <x> <y> <radius>. If 0, try to "
                 "read from video")
      ->check(CLI::NonNegativeNumber);

  app.add_flag("--draw", draw, "show video and detections")
      ->configurable(false);
  app.add_flag("--export-images", export_images,
               "export an image of each particle")
      ->configurable(false);
  app.add_flag("--export-hdf5", export_hdf5,
               "export VTK compatible HDF5 data sets for each particle")
      ->configurable(false);
  app.set_version_flag("--version,-v", CMAKE_PROJECT_VERSION,
                       "display version and exit");

  auto filter_cmd =
      app.add_subcommand("filter", "options for filtering particles");
  filter_cmd->configurable();
  filter_cmd
      ->add_option("--area", contour_filter_args.area, "allowed particle area")
      ->check(CLI::NonNegativeNumber);
  filter_cmd
      ->add_option("--aspect", contour_filter_args.aspect,
                   "allowed particle aspect ratio")
      ->check(CLI::Range(0.0, 1.0));
  filter_cmd
      ->add_option("--circularity", contour_filter_args.circularity,
                   "allowed particle circularity")
      ->check(CLI::Range(0.0, 1.0));
  filter_cmd
      ->add_option("--convexity", contour_filter_args.convexity,
                   "allowed particle convexity")
      ->check(CLI::Range(0.0, 1.0));
  filter_cmd
      ->add_option("--intensity", contour_filter_args.intensity,
                   "allowed particle intensity (darkness)")
      ->check(CLI::NonNegativeNumber);
  filter_cmd
      ->add_option("--radius", contour_filter_args.radius,
                   "allowed particle radius")
      ->check(CLI::NonNegativeNumber);
  filter_cmd
      ->add_option("--sharpness", contour_filter_args.sharpness,
                   "allowed particle sharpness")
      ->check(CLI::NonNegativeNumber);

  app.add_flag_callback(
         "--create-config",
         [&]() {
           create_config = true;
           file_opt->get_validator("FileExists")->active(false);
           file_opt->check(CLI::NonexistentPath);
         },
         "write default values to a new config file at 'file'")
      ->configurable(false)
      ->callback_priority(CLI::CallbackPriority::First);
  app.set_config("--config", std::string(), "read options from a config file");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  if (create_config) {
    std::ofstream cfs(CLI::to_path(inname));
    std::cout << "deault config file created at " << inname << ", exiting"
              << std::endl;
    cfs << app.config_to_str(true);
    return 0;
  }

  // Convert some of the parsed options
  std::filesystem::path path(CLI::to_path(inname));
  std::filesystem::path output_dir;
  if (outname.empty()) {
    output_dir = path.parent_path() / "processed";
  } else {
    output_dir = CLI::to_path(outname);
  }

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

  std::filesystem::create_directory(output_dir);
  auto image_dir = output_dir / "particle_images";
  auto hdf5_dir = output_dir / "particle_hdf5";
  if (export_images) {
    std::filesystem::create_directory(image_dir);
  }
  if (export_hdf5) {
    std::filesystem::create_directory(hdf5_dir);
  }

  // just used for date
  auto start_time = std::chrono::system_clock::now();

  std::ofstream results_output(
      output_dir /
          (std::format("{0:%H}_{0:%M}_{0:%OS}", start_time) + "_particles.csv"),
      std::ios::out);
  write_particle_header(results_output);

  // load a frame and find the ROI
  cv::UMat frame, capillary_mask;
  cap.read(frame);
  capillary_mask = cv::UMat::zeros(frame.rows, frame.cols, CV_8U);

  std::cout << "Processsing " << path << std::endl;

  std::cout << "\tframes = " << frame_count << std::endl;
  std::cout << "\tsize = " << width << " x " << height << std::endl;

  // If radius is zero, try to find from image
  if (capillary[2] == 0.f) {
    capillary = find_capillary(frame);
  }
  if (capillary[2] == 0.f) {
    std::cerr << "\tcould not detect capillary" << std::endl;
    return 1;
  } else {
    std::cout << "\tcapillary detected at " << capillary[0] << " x "
              << capillary[1] << " with radius " << capillary[2] << " px"
              << std::endl;
    // shrink
    capillary[2] *= 0.95;
  }
  cv::circle(capillary_mask, cv::Point(capillary[0], capillary[1]),
             capillary[2], 255, -1);

  // setup arrays
  cv::UMat acc_mean;
  cv::UMat acc_var = cv::UMat::zeros(frame.rows, frame.cols, CV_32F);

  // init the accumulated mean and variance
  frame.convertTo(acc_mean, CV_32F);

  // begin by reading the required number of frames to predict the background
  init_background(cap, acc_mean, acc_var, background_frames);

  if (draw) {
    cv::namedWindow("frame", cv::WINDOW_NORMAL);
    // cv::namedWindow("diff", cv::WINDOW_NORMAL);
  };

  // reset the video
  int frame_pos = 0;
  cap.set(cv::CAP_PROP_POS_FRAMES, frame_pos);
  start_time = std::chrono::system_clock::now();

  int particle_count = 0;
  std::vector<Particle> particles;

  cv::UMat processed, threshold;

  while (frame_pos < frame_count) {
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

    preprocess_and_threshold(frame, acc_mean, acc_var, processed, threshold,
                             zscore, unsharp_alpha);

    cv::bitwise_and(threshold, capillary_mask, threshold);
    // get contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(threshold, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    // move the diff image to the CPU, since it gets cropped and is then faster
    cv::Mat cpu_proc = processed.getMat(cv::ACCESS_READ);
    // only move frame image if we are exporting, expensive operation
    cv::Mat cpu_frame;
    if (export_images)
      cpu_frame = frame.getMat(cv::ACCESS_READ);

    filter_contours(contours, cpu_proc, contour_filter_args);

    // update_particles(particles, contours);
    std::for_each(
        contours.begin(), contours.end(),
        [&](const std::vector<cv::Point> &contour) {
          bool existing = false;
          for (auto &particle : particles) {
            // check circles first
            double dist = contour_box_distance(contour, particle.contour());
            if (dist > particle_distance)
              continue;
            // finer check for close particles
            dist = contour_distance(contour, particle.contour());
            if (dist < particle_distance) {
              particle.update(frame_pos, contour, cpu_proc, cpu_frame);
              existing = true;
              break;
            }
          }
          if (!existing) {
            particles.push_back(
                Particle(frame_pos, contour, cpu_proc, cpu_frame));
            particle_count += 1;
          }
        });

    // remove untrakced (old) particles from the vector
    auto pivot = std::stable_partition(
        particles.begin(), particles.end(), [&](const Particle &p) {
          return frame_pos - p.lastFrame() < particle_frames;
        });
    std::vector<Particle> output_particles(
        std::make_move_iterator(pivot),
        std::make_move_iterator(particles.end()));
    particles.erase(pivot, particles.end());

    // create a color image and draw the contuors
    if (draw) {
      cv::UMat rgb_frame;
      draw_particles_on_frame(frame, rgb_frame, particles);
      // draw capilary bounds
      cv::circle(rgb_frame, cv::Point(capillary[0], capillary[1]), capillary[2],
                 cv::Scalar(0, 255, 0), 1);
      cv::imshow("frame", rgb_frame);

      int key = cv::waitKey(20);
      if (key == 'q') {
        break;
      }
    }

    // output the particles
    write_particle_data(output_particles, results_output);

    if (export_images) {
      for (const auto &p : output_particles) {
        auto image_path = image_dir / std::to_string(p.id()).append(".png");
        if (save_particle_image(p, image_path))
          return 1;
      }
    }

    if (export_hdf5) {
      for (const auto &p : output_particles) {
        auto h5_path = hdf5_dir / std::to_string(p.id()).append(".vtkhdf");
        if (save_particle_data_hdf5(p, h5_path))
          return 1;
      }
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
      std::cout << particles.size() << " active, ";
      std::cout << std::format("{:%T}", remaining) << " remaining.\r"
                << std::flush;
    }

    frame_pos++;
  } // while
  // export any remaining particles
  write_particle_data(particles, results_output);

  if (export_images) {
    for (const auto &p : particles) {
      auto image_path = image_dir / std::to_string(p.id()).append(".png");
      if (save_particle_image(p, image_path))
        return 1;
    }
  }

  if (export_hdf5) {
    for (const auto &p : particles) {
      auto h5_path = hdf5_dir / std::to_string(p.id()).append(".vtkhdf");
      if (save_particle_data_hdf5(p, h5_path))
        return 1;
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
