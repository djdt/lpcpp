#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "particle.hpp"

bool mask_capillary(cv::InputArray &input, cv::Mat &mask, double &um_per_px,
                    const double capillary_diameter = 750.0);

void unsharp_mask(const cv::cuda::GpuMat &image, cv::cuda::GpuMat &output,
                  double alpha);

void update_background(const cv::Mat &frame, cv::Mat &mean, cv::Mat &var,
                       int pos);

bool init_background(cv::VideoCapture &cap, cv::Mat &mean, cv::Mat &var,
                     int frame_count);

void find_particles(const cv::Mat &frame, const cv::Mat &mean,
                    const cv::Mat &var, const double zscore,
                    const cv::Mat &mask, std::vector<Particle> &particles,
                    const int current_frame, int current_id);
