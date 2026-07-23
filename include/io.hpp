#pragma once

#include "particle.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

// Writes the column names used in `write_particle_data` to a file
void write_particle_properties_header(std::ofstream &ofs);
// Exports particle parameters to a file
void write_particle_properties(const std::vector<Particle> &particles,
                               std::ofstream &ofs);

bool save_particle_contours(const Particle &particle,
                            const std::filesystem::path &path);

// Saves a particles mask, processed and raw images to a png
bool save_particle_data_png(const Particle &particle,
                            const std::filesystem::path &path);

// Saves the particles contour masks and processed images to a VTK ImageData
// formatted text file
bool save_particle_data_vtk(const Particle &particle,
                            const std::filesystem::path &path);

// Saves the particles contour masks and processed images to an VTK compatible
// compressed HDF5 archive
bool save_particle_data_hdf5(const Particle &particle,
                             const std::filesystem::path &path);
