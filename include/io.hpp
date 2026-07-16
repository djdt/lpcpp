#pragma once

#include <filesystem>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "particle.hpp"

void write_particle_header(std::ofstream &ofs);
void write_particle_data(const std::vector<Particle> &particles,
                         std::ofstream &ofs);
// bool write_particle_images(const std::vector<Particle> &particles,
//                            const std::filesystem::path &output_dir);

bool save_particle_image(const Particle &particle,
                         const std::filesystem::path &path);

void draw_particles_on_frame(cv::InputArray &input,
                             cv::InputOutputArray &output,
                             std::vector<Particle> &particles);
