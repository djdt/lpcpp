#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "particle.hpp"

void unsharp_mask(const cv::cuda::GpuMat &image, cv::cuda::GpuMat &output,
                  double alpha);

void update_background(const cv::cuda::GpuMat &frame, cv::cuda::GpuMat &mean,
                       cv::cuda::GpuMat &var, int pos);

bool init_background(cv::VideoCapture &cap, cv::cuda::GpuMat &mean,
                     cv::cuda::GpuMat &var, int frame_count);

void find_particles(const cv::cuda::GpuMat &frame, const cv::cuda::GpuMat &mean,
                    const cv::cuda::GpuMat &var, const double zscore,
                    const cv::cuda::GpuMat &mask,
                    std::vector<Particle> &particles, const int current_frame,
                    int current_id);
