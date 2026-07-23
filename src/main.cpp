#include "CLI11.hpp"

#include "contours.hpp"
#include "cpuproc.hpp"
#include "io.hpp"
#include "particle.hpp"
#include "util.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#ifndef CMAKE_PROJECT_VERSION
#define CMAKE_PROJECT_VERSION "0.0.0"
#endif

void draw_particles_on_frame(cv::InputArray &input,
                             cv::InputOutputArray &output,
                             std::vector<Particle> &particles) {
  output.createSameSize(input, CV_8UC3);
  cv::cvtColor(input, output, cv::COLOR_GRAY2BGR);

  auto color = cv::Scalar(0, 0, 255);
  std::vector<std::vector<cv::Point>> contours;
  contours.reserve(particles.size());

  std::transform(particles.begin(), particles.end(),
                 std::back_inserter(contours),
                 [](const Particle &p) { return p.contour(); });

  cv::drawContours(output, contours, -1, color, 1.0, 8);
}

bool export_particle_data(const std::filesystem::path &path,
                          const ::std::vector<Particle> &particles,
                          bool png = false, bool tiff = false, bool vti = false,
                          bool hdf5 = false) {
  if (png) {
    std::filesystem::create_directory(path / "png");
    for (const auto &p : particles) {
      auto image_path = path / "png" / std::to_string(p.id()).append(".png");
      if (save_particle_data_png(p, image_path))
        return true;
    }
  }
  if (vti) {
    std::filesystem::create_directory(path / "vti");
    for (const auto &p : particles) {
      auto vtk_path = path / "vti" / std::to_string(p.id()).append(".vti");
      if (save_particle_data_vtk(p, vtk_path))
        return true;
    }
  }
  if (hdf5) {
    std::filesystem::create_directory(path / "hdf5");
    for (const auto &p : particles) {
      auto hdf5_path = path / "hdf5" / std::to_string(p.id()).append(".vtkhdf");
      if (save_particle_data_hdf5(p, hdf5_path))
        return true;
    }
  }

  return false;
}

int main(int argc, char *argv[]) {

  CLI::App app;
  app.option_defaults()->always_capture_default();
  app.set_help_flag("");
  app.set_help_all_flag("-h,--help");

  std::string inname, outname;

  int background_frames = 1000;
  int particle_frames = 10;

  double particle_distance = 5.0;
  double zscore = 3.0;
  double unsharp_alpha = 1.0;

  std::array<float, 3> capillary = {0.f, 0.f, 0.f};

  bool create_config = false;
  bool draw = false;
  bool export_png = false, export_hdf5 = false, export_vti = false;

  ParticleFrameMetric selection_metric = METRIC_CENTER_WEIGHTED_INTENSITY;
  std::map<std::string, ParticleFrameMetric> metric_map = {
      {"averageIntensity", METRIC_AVERAGE_INTENSITY},
      {"centralIntensity", METRIC_CENTER_WEIGHTED_INTENSITY},
      {"sharpness", METRIC_SHARPNESS}};
  PreprocessImageMode preprocess_mode = PROC_MODE_INVERT;
  std::map<std::string, PreprocessImageMode> mode_map = {
      {"light", PROC_MODE_NORMAL},
      {"dark", PROC_MODE_INVERT},
      {"absolute", PROC_MODE_ABSOLUTE}};

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

  app.add_option("--selection-metric", selection_metric,
                 "method of selecting the particle frame for processing")
      ->transform(CLI::CheckedTransformer(metric_map, CLI::ignore_case));
  app.add_option("--detection-mode", preprocess_mode,
                 "method of thresholding differences from the background")
      ->transform(CLI::CheckedTransformer(mode_map, CLI::ignore_case));
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
#ifdef ENABLE_HDF5_EXPORT
  app.add_flag("--export-hdf5", export_hdf5,
               "export VTK compatible HDF5 data sets for each particle")
      ->configurable(false);
#endif // ENABLE_HDF5_EXPORT
  app.add_flag("--export-png", export_png,
               "export a PNG image for each particle")
      ->configurable(false);
  app.add_flag("--export-vti", export_vti,
               "export a VTK ImageDara for each particle")
      ->configurable(false);

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
                   "allowed particle diameter")
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
  app.set_version_flag("--version,-v", CMAKE_PROJECT_VERSION,
                       "display version and exit");

  //
  // parse the arguments
  //

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  // early exit if create config is used
  if (create_config) {
    std::ofstream cfs(CLI::to_path(inname));
    std::cout << "deault config file created at " << inname << ", exiting"
              << std::endl;
    cfs << app.config_to_str(true);
    return 0;
  }

  // Convert some of the parsed options
  std::filesystem::path path(CLI::to_path(inname));
  std::filesystem::path output_dir(CLI::to_path(outname));
  if (output_dir.empty())
    output_dir = path.parent_path() / "processed";

  //
  // open video capture and read some props
  //

  auto cap = cv::VideoCapture(path.string(), cv::CAP_FFMPEG);
  if (!cap.set(cv::CAP_PROP_CONVERT_RGB, 0)) {
    std::cerr << "cannot read as greyscale" << std::endl;
    return 1;
  }
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  int frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);

  // info message
  std::cout << "Processsing " << path << std::endl;
  std::cout << "\tframes = " << frame_count << std::endl;
  std::cout << "\tsize = " << width << " x " << height << std::endl;

  //
  // detect capillary and mask off bounds
  //

  // load a frame and find the ROI
  cv::UMat frame, capillary_mask;
  cap.read(frame);
  capillary_mask = cv::UMat::zeros(frame.rows, frame.cols, CV_8U);

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

  //
  // initialise the background and variance accumulated arrays
  //

  // init the accumulated mean and variance
  cv::UMat acc_mean;
  cv::UMat acc_var = cv::UMat::zeros(frame.rows, frame.cols, CV_32F);
  frame.convertTo(acc_mean, CV_32F);

  // begin by reading the required number of frames to predict the background
  if (init_background(cap, acc_mean, acc_var, background_frames)) {
    return 1;
  }

  if (draw) {
    cv::namedWindow("frame", cv::WINDOW_NORMAL);
  };

  //
  // create output directories and files
  //

  // create output directory
  std::filesystem::create_directory(output_dir);
  // open results file and populate with header
  std::ofstream ofs_results(
      output_dir /
      (std::format("{0:%H}_{0:%M}_{0:%OS}", std::chrono::system_clock::now()) +
       "_particles.csv"));
  write_particle_properties_header(ofs_results);

  //
  // start the main loop
  //

  cv::UMat processed, threshold;
  std::vector<Particle> particles;
  int particle_count = 0;

  auto start_time = std::chrono::system_clock::now();
  // time interval for next update of progress
  auto update_time = start_time + std::chrono::seconds(1);

  // reset the video
  int frame_pos = 0;
  cap.set(cv::CAP_PROP_POS_FRAMES, frame_pos);

  while (frame_pos < frame_count) {
    // read frame and exit if end of video
    cap.read(frame);
    if (frame.empty()) {
      break;
    }

    // update the background accumulated mean and variance,
    // if on an frame not processed during init
    if (frame_pos > background_frames) {
      update_background(frame, acc_mean, acc_var, frame_pos);
    }

    //
    // preprocessing and threshold
    //

    // median blur first, faster as CV_8U
    cv::medianBlur(frame, frame, 5);
    frame.convertTo(processed, CV_32F); // ensure type correct for preproc

    // main processing, extract the difference from the mean, apply unsharp mask
    // then get threshold where greater than then z * std
    preprocess_and_threshold(processed, acc_mean, acc_var, processed, threshold,
                             zscore, unsharp_alpha, preprocess_mode);
    // mask off the capillary edges
    cv::bitwise_and(threshold, capillary_mask, threshold);

    //
    // find and filter contours
    //

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(threshold, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    filter_contours(contours, processed, contour_filter_args);

    //
    // update particles with new contours
    //

    // move the diff image to the CPU, since it gets cropped and is then faster
    cv::Mat cpu_proc = processed.getMat(cv::ACCESS_READ);
    // only move frame image if we are exporting, expensive operation
    cv::Mat cpu_frame;
    if (export_png)
      cpu_frame = frame.getMat(cv::ACCESS_READ);

    // update_particles(particles, contours);
    std::for_each(
        contours.begin(), contours.end(),
        [&](const std::vector<cv::Point> &contour) {
          bool existing = false;
          for (auto &particle : particles) {
            cv::Rect rect = cv::boundingRect(contour);
            cv::Rect particle_rect = cv::boundingRect(particle.contour());
            // check boxes first, early exit if far
            double dist = box_edge_distance(rect, particle_rect);
            if (dist > particle_distance)
              continue;

            // finer check for close particles, larger contour as first
            if (rect.size().area() > particle_rect.size().area()) {
              dist = contour_edge_distance(contour, particle.contour());
            } else {
              dist = contour_edge_distance(particle.contour(), contour);
            }

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

    //
    // remove untrakced (old) particles from the vector
    //

    auto pivot = std::stable_partition(
        particles.begin(), particles.end(), [&](const Particle &p) {
          return frame_pos - p.lastFrame() < particle_frames;
        });
    std::vector<Particle> output_particles(
        std::make_move_iterator(pivot),
        std::make_move_iterator(particles.end()));
    particles.erase(pivot, particles.end());

    //
    // draw a color image and contuors
    //

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

    //
    // output
    //

    write_particle_properties(output_particles, ofs_results);
    export_particle_data(output_dir, output_particles, export_png);

    //
    // update progress
    //

    auto frame_time = std::chrono::system_clock::now();
    if (frame_time > update_time || frame_pos == frame_count) {
      double fps;
      auto remaining = get_remaining_time(start_time, frame_time, frame_pos,
                                          frame_count, fps);

      std::cout << "\t...processing :: frame " << frame_pos << "/"
                << frame_count << " @ ";
      std::cout << std::setw(3) << static_cast<int>(fps) << " FPS, ";
      std::cout << particle_count << " particles, ";
      std::cout << particles.size() << " active, ";
      std::cout << std::format("{:%T}", remaining) << " remaining.\r"
                << std::flush;

      update_time = frame_time + std::chrono::seconds(1);
    }

    frame_pos++;
  } // while

  //
  // export any remaining particles and images
  //

  write_particle_properties(particles, ofs_results);
  export_particle_data(output_dir, particles, export_png);

  cv::UMat acc_out, acc_var_out;
  acc_mean.convertTo(acc_out, CV_8U);
  cv::imwrite((output_dir / "background_mean.png").string(), acc_out);
  acc_var.convertTo(acc_var_out, CV_8U);
  cv::imwrite((output_dir / "background_var.png").string(), acc_var_out);

  //
  // print total time
  //

  auto total_duration = std::chrono::duration<double>(
      std::chrono::system_clock::now() - start_time);

  std::cout << std::endl
            << "Finished in " << std::format("{:%T}", total_duration)
            << std::endl;

  return 0;
}
