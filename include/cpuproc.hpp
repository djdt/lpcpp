#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "particle.hpp"

cv::Vec3f find_capillary(cv::InputArray &input);

void unsharp_mask(cv::InputArray &image, cv::OutputArray &output, double alpha);

void update_background(cv::InputArray &frame, cv::InputOutputArray &mean,
                       cv::InputOutputArray &var, int pos);

bool init_background(cv::VideoCapture &cap, cv::InputOutputArray &mean,
                     cv::InputOutputArray &var, int frame_count);

void find_particles(cv::InputArray &frame, cv::InputArray &mean,
                    cv::InputArray &var, const double zscore,
                    cv::InputArray &mask, const double unsharp_alpha,
                    std::vector<Particle> &particles, const int current_frame);

/* Filters particles by property, removing failing from the vector.
 * To enabled a filter pass different min and max values. */
void filter_particles(std::vector<Particle> &particles, struct filter_args);

/* Filters particles in that already exist in old_particles, by edge distance.
 * A comparison is used to choose which of the new or old particles is kept. */
void filter_existing_particles(
    std::vector<Particle> &old_particles, std::vector<Particle> &new_particles,
    const std::function<bool(const Particle &, const Particle &)> comparision,
    const double edge_distance = 20.0);
