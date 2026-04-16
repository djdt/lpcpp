#pragma once

#include <execution>
#include <filesystem>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "particle.hpp"

void write_particle_header(std::ofstream &ofs);
void write_particle_data(const std::vector<Particle> &particles,
                         std::ofstream &ofs);
bool write_particle_images(const std::vector<Particle> &particles,
                           const std::filesystem::path &output_dir);

void read_filter_config(std::string path, filter_args &args);

template <typename Iter>
void draw_particles_on_frame(cv::InputArray &input,
                             cv::InputOutputArray &output, Iter rbegin,
                             Iter rend, const int particle_frames) {

  output.createSameSize(input, CV_8UC3);
  cv::cvtColor(input, output, cv::COLOR_GRAY2BGR);

  auto color = cv::Scalar(0, 0, 255);
  int decay = 255 / particle_frames;
  std::vector<std::vector<cv::Point>> contours;

  for (auto it = rbegin; it != rend; ++it) {
    contours.resize(it->size());
    std::transform(std::execution::par, it->begin(), it->end(),
                   contours.begin(),
                   [](const Particle &p) { return p.contour(); });

    cv::drawContours(output, contours, -1, color, 1.0, 8);
    color[2] -= decay;
  }
}
