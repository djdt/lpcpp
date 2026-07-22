#pragma once

#include "particle.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

void draw_particles_on_frame(cv::InputArray &input,
                             cv::InputOutputArray &output,
                             std::vector<Particle> &particles);

void write_particle_header(std::ofstream &ofs);
void write_particle_data(const std::vector<Particle> &particles,
                         std::ofstream &ofs);
// bool write_particle_images(const std::vector<Particle> &particles,
//                            const std::filesystem::path &output_dir);

bool save_particle_image(const Particle &particle,
                         const std::filesystem::path &path);

bool save_particle_data_vtk(const Particle &particle,
                            const std::filesystem::path &path);

bool save_particle_data_hdf5(const Particle &particle,
                             const std::filesystem::path &path);
