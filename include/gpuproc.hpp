#include <opencv2/core.hpp>
#include <opencv2/cudafilters.hpp>

#include "asynccapture.hpp"
#include "particle.hpp"

void unsharp_mask(const cv::cuda::GpuMat &image, cv::cuda::GpuMat &output,
                  cv::cuda::Filter *sx, cv::cuda::Filter *sy,
                  double alpha = 1.0);
void update_background(const cv::cuda::GpuMat &frame, cv::cuda::GpuMat &mean,
                       cv::cuda::GpuMat &var, int pos);

bool init_background(AsyncVideoCapture &cap, cv::cuda::GpuMat &mean,
                     cv::cuda::GpuMat &var, int frame_count);

void find_particles(const cv::cuda::GpuMat &frame, const cv::cuda::GpuMat &mean,
                    const cv::cuda::GpuMat &var, const double zscore,
                    const cv::cuda::GpuMat &mask,
                    std::vector<Particle> &particles, const int current_frame,
                    int current_id);
