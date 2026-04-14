#include <iostream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/videoio.hpp>

#include "median.cuh"
#include "particle.hpp"
#include "util.hpp"

const auto sobelx = cv::cuda::createSobelFilter(CV_32F, CV_32F, 1, 0, 3);
const auto sobely = cv::cuda::createSobelFilter(CV_32F, CV_32F, 0, 1, 3);
// const auto erode = cv::cuda::createMorphologyFilter(
//     cv::MORPH_ERODE, CV_8U,
//     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

void unsharp_mask(const cv::cuda::GpuMat &image, cv::cuda::GpuMat &output,
                  double alpha = 1.0) {
  cv::cuda::GpuMat sx, sy, mag;
  sobelx->apply(image, sx);
  sobely->apply(image, sy);
  cv::cuda::magnitude(sx, sy, mag);
  cv::cuda::addWeighted(image, 1.0 + alpha, mag, -alpha, 0, output);
}

void update_background(const cv::cuda::GpuMat &frame, cv::cuda::GpuMat &mean,
                       cv::cuda::GpuMat &var, int pos) {
  double weight = 1.0 / static_cast<double>(pos);

  cv::cuda::GpuMat frame_var;
  frame.convertTo(frame_var, CV_32F);

  cv::cuda::addWeighted(frame_var, weight, mean, 1.0 - weight, 0.0, mean);

  cv::cuda::subtract(frame_var, mean, frame_var);
  cv::cuda::pow(frame_var, 2.0, frame_var);

  cv::cuda::addWeighted(frame_var, weight, var, 1.0 - weight, 0.0, var);
}

bool init_background(cv::VideoCapture &cap, cv::cuda::GpuMat &mean,
                     cv::cuda::GpuMat &var, int frame_count) {
  int frame_pos = 0;
  cap.set(cv::CAP_PROP_POS_FRAMES, frame_pos);
  // cap.invalidate();

  cv::Mat cpu_frame;
  cv::cuda::GpuMat frame;
  auto stream = cv::cuda::Stream();

  auto start_time = std::chrono::system_clock::now();

  // read in first frame
  cap.read(cpu_frame);
  frame.upload(cpu_frame, stream);

  while (frame_pos++ < frame_count) {
    stream.waitForCompletion();

    cap.read(cpu_frame);
    frame.upload(cpu_frame, stream);

    if (frame.empty()) {
      std::cerr << "video does not contain enough background frames"
                << std::endl;
      return true;
    }
    // cv::cuda::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

    // update the background accumulated mean and variance
    update_background(frame, mean, var, frame_pos);

    // update progress
    if (frame_pos % 100 == 0) {
      double fps;
      auto remaining =
          get_remaining_time(start_time, frame_pos, frame_pos, fps);

      std::cout << "\t...processing background :: frame " << frame_pos << "/"
                << frame_count << " @ ";
      std::cout << std::setw(3) << static_cast<int>(fps) << " FPS, ";
      std::cout << std::format("{:%T}", remaining) << " remaining.\r"
                << std::flush;
    }
  }
  std::cout << std::endl;
  return false;
}

void find_particles(const cv::cuda::GpuMat &frame, const cv::cuda::GpuMat &mean,
                    const cv::cuda::GpuMat &var, const double zscore,
                    const cv::cuda::GpuMat &mask, const double unsharp_alpha,
                    std::vector<Particle> &particles, const int current_frame,
                    int current_id) {

  cv::Mat cpu_diff, cpu_thresh;
  cv::cuda::GpuMat diff;
  // calculate the difference between frame and mean
  frame.convertTo(diff, CV_32F);
  cv::cuda::subtract(diff, mean, diff);
  cv::cuda::multiplyWithScalar(diff, -1, diff);

  // median blur
  medianFilter3x3(diff, diff);

  // sharpen
  if (unsharp_alpha > 0.0)
    unsharp_mask(diff, diff, unsharp_alpha);

  // mask differences below x std deviations
  cv::cuda::GpuMat std;
  cv::cuda::sqrt(var, std);
  cv::cuda::multiplyWithScalar(std, zscore, std);

  cv::cuda::GpuMat thresh = cv::cuda::GpuMat(frame.rows, frame.cols, CV_8U);
  cv::cuda::compare(diff, std, thresh, cv::CMP_GT);

  cv::cuda::bitwise_and(thresh, mask, thresh);

  thresh.download(cpu_thresh);
  diff.download(cpu_diff);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(cpu_thresh, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);
  particles.reserve(contours.size());
  std::transform(
      contours.begin(), contours.end(), std::back_inserter(particles),
      [&](const std::vector<cv::Point> &contour) {
        return Particle(contour, cpu_diff, current_frame, current_id++);
      });
}
