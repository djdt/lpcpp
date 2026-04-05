#include <filesystem>
#include <fstream>
#include <vector>

#include "particle.hpp"

void write_particle_header(std::ofstream &ofs);
void write_particle_data(const std::vector<Particle> &particles,
                         std::ofstream &ofs);
bool write_particle_images(const std::vector<Particle> &particles,
                           const std::filesystem::path &output_dir);

void read_filter_config(std::string path, filter_args &args);
