#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "particle.hpp"

bool mask_capillary(cv::InputArray &input, cv::InputOutputArray &mask,
                    double &um_per_px, const double capillary_diameter = 750.0);

void unsharp_mask(cv::InputArray &image, cv::OutputArray &output, double alpha);

void update_background(cv::InputArray &frame, cv::InputOutputArray &mean,
                       cv::InputOutputArray &var, int pos);

bool init_background(cv::VideoCapture &cap, cv::InputOutputArray &mean,
                     cv::InputOutputArray &var, int frame_count);

void find_particles(cv::InputArray &frame, cv::InputArray &mean,
                    cv::InputArray &var, const double zscore,
                    cv::InputArray &mask, const double unsharp_alpha,
                    std::vector<Particle> &particles, const int current_frame,
                    int current_id);
