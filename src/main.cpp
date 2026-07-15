#include <algorithm>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "CLI11.hpp"

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

  std::string inname;
  std::string outname;
  std::string confname;

  int background_frames = 1000;
  int particle_frames = 50;
  double particle_distance = 3.0;
  double zscore = 3.0;
  double unsharp = 1.0;
  std::array<float, 3> capillary = {0.f, 0.f, 0.f};

  bool draw = false;
  bool export_images = false;

  filter_args contour_filter_args;

  app.add_option("file", inname, "path to the captured OIM video")
      ->required()
      ->check(CLI::ExistingFile)
      ->configurable(false);
  app.add_option("--output,-o", outname,
                 "specify the output directory, defaults to 'processed'")
      ->check(CLI::NonexistentPath | CLI::ExistingDirectory)
      ->configurable(false);

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
  app.add_option("--unsharp", unsharp, "alpha value of the unsharp mask")
      ->check(CLI::NonNegativeNumber);
  app.add_option("--capillary", capillary,
                 "capillary position and radius <x> <y> <radius>. If 0, try to "
                 "read from video")
      ->check(CLI::NonNegativeNumber);

  app.add_flag("--draw", draw, "show video and detections")
      ->configurable(false);
  ;
  app.add_flag("--export-images", export_images,
               "export an image of each particle")
      ->configurable(false);
  ;
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

  app.set_config("--config", std::string(),
                 "read options from a config file, if no path is passed "
                 "creates a default config");

  try {
    app.parse(argc, argv);
  } catch (const CLI::FileError &e) {
    std::string conf = app.get_config_ptr()->as<std::string>();
    std::cout << "writing default config to " << conf << std::endl;
    std::ofstream cfs(CLI::to_path(conf));
    cfs << app.config_to_str(true);
    return 0;
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
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
  auto image_dir = output_dir / "particles";
  if (export_images) {
    std::filesystem::create_directory(image_dir);
  }

  // just used for date
  auto start_time = std::chrono::system_clock::now();

  std::ofstream results_output(
      output_dir /
          (std::format("{0:%H}_{0:%M}_{0:%OS}", start_time) + "_particles.csv"),
      std::ios::out);
  write_particle_header(results_output);

  // load a frame and find the ROI
  cv::UMat frame, diff, mask;
  cap.read(frame);
  mask = cv::UMat::zeros(frame.rows, frame.cols, CV_8U);

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
  cv::circle(mask, cv::Point(capillary[0], capillary[1]), capillary[2], 255,
             -1);

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

  // init the particle vars
  // std::deque<std::vector<Particle>> particles;
  // int particle_count = 0;
  std::vector<Particle> particles;

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

    // get the differenct from the mean
    cv::subtract(frame, acc_mean, diff, cv::noArray(), CV_32F);
    cv::multiply(diff, -1.f, diff);

    // apply median blur for small defects
    cv::medianBlur(diff, diff, 5);

    // sharpen image to reduce particle edge blur
    if (unsharp > 0.0)
      unsharp_mask(diff, diff, unsharp);

    // threshold using std and mean
    cv::UMat threshold;
    niblack_threshold(diff, acc_mean, acc_var, threshold, zscore);
    cv::bitwise_and(threshold, mask, threshold);

    // get contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(threshold, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    // move the diff image to the CPU, since it gets cropped and is then faster
    cv::Mat cpu_diff = diff.getMat(cv::ACCESS_READ);
    // only move frame image if we are exporting, expensive operation
    cv::Mat cpu_frame;
    if (export_images)
      cpu_frame = frame.getMat(cv::ACCESS_READ);

    filter_contours(contours, cpu_diff, contour_filter_args);

    // update_particles(particles, contours);
    std::for_each(
        contours.begin(), contours.end(),
        [&](const std::vector<cv::Point> &contour) {
          bool existing = false;
          for (auto &particle : particles) {
            double dist =
                cv::pointPolygonTest(contour, particle.center(), true);
            if (dist < particle_distance) {
              particle.update(contour, cpu_diff, frame_pos, cpu_frame);
              existing = true;
              break;
            }
          }
          if (!existing) {
            particles.push_back(
                Particle(contour, cpu_diff, frame_pos, cpu_frame));
          }
        });

    // filter particle based on parameters
    // filter_particles(new_particles, contour_filter_args);
    // filter based on last n frames

    // for (auto it = particles.begin(); it != particles.end(); ++it) {
    //   filter_existing_particles(
    //       *it, new_particles,
    //       [](const Particle &a, const Particle &b) {
    //         return a.centerWeightedIntensity() > b.centerWeightedIntensity();
    //       },
    //       particle_distance);
    // }
    //
    // // add a raw image to each particle, slow so only if images needed
    // if (export_images) {
    //   cv::Mat cpu_frame = frame.getMat(cv::ACCESS_READ);
    //   std::for_each(new_particles.begin(), new_particles.end(),
    //                 [&cpu_frame](Particle &p) { p.setRawImage(cpu_frame); });
    // }
    // particles.push_back(new_particles);

    // create a color image and draw the contuors
    if (draw) {
      cv::UMat rgb_frame;
      draw_particles_on_frame(frame, rgb_frame, particles.rbegin(),
                              particles.rend(), particle_frames);
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
    if (particles.size() > particle_frames) {
      auto output_particles = particles.front();

      write_particle_data(output_particles, results_output);
      if (export_images) {
        if (write_particle_images(output_particles, image_dir)) {
          return 1;
        }
      }
      particle_count += particles[0].size();
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

    frame_pos++;
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
