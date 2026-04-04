#pragma once

#include <filesystem>
#include <fstream>
#include <vector>

#include "particle.hpp"

void write_particle_header(std::ofstream &ofs);
void write_particle_data(const std::vector<Particle> &particles,
                         std::ofstream &ofs);
bool export_particle_images(const std::vector<Particle> &particles,
                            const std::filesystem::path &output_dir);

template <typename Iter>
void draw_current_frame(cv::InputArray &frame, const Iter &particles_begin,
                        const Iter &particles_end);
