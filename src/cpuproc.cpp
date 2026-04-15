#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "particle.hpp"
#include "util.hpp"

bool mask_capillary(cv::InputArray &input, cv::InputOutputArray &mask,
                    double &um_per_px,
                    const double capillary_diameter = 750.0) {
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(input, circles, cv::HOUGH_GRADIENT, 1.0,
                   static_cast<float>(input.rows()) / 2.f, 50, 5,
                   input.rows() / 4, input.rows());

  if (circles.size() == 0) {
    return true;
  } else {
    std::cout << "\tcapillary detected at " << circles[0][0] << " x "
              << circles[0][1] << " with radius " << circles[0][2] << std::endl;
  }

  um_per_px = capillary_diameter / (2.0 * circles[0][2]);

  mask.createSameSize(input, CV_8U);
  mask.setTo(0);
  cv::circle(mask, cv::Point(circles[0][0], circles[0][1]), circles[0][2] * 0.9,
             255, -1);
  return false;
}

void unsharp_mask(cv::InputArray &image, cv::OutputArray &output,
                  double alpha = 1.0) {
  cv::UMat sobelx, sobely, mag;
  cv::Sobel(image, sobelx, CV_32F, 1, 0, 3);
  cv::Sobel(image, sobely, CV_32F, 0, 1, 3);
  cv::magnitude(sobelx, sobely, mag);
  cv::addWeighted(image, 1.0 + alpha, mag, -alpha, 0, output);
}

void update_background(cv::InputArray &frame, cv::InputOutputArray &mean,
                       cv::InputOutputArray &var, int pos) {
  double weight = 1.0 / static_cast<double>(pos);

  cv::addWeighted(frame, weight, mean, 1.0 - weight, 0.0, mean, CV_32F);

  cv::UMat tmp;
  cv::subtract(frame, mean, tmp, cv::noArray(), CV_32F);
  cv::pow(tmp, 2.0, tmp);
  cv::addWeighted(tmp, weight, var, 1.0 - weight, 0.0, var);
}

bool init_background(cv::VideoCapture &cap, cv::InputOutputArray &mean,
                     cv::InputOutputArray &var, int frame_count) {
  int frame_pos = 0;
  cap.set(cv::CAP_PROP_POS_FRAMES, frame_pos);

  cv::UMat frame;

  auto start_time = std::chrono::system_clock::now();

  while (frame_pos++ < frame_count) {
    cap.read(frame);
    if (frame.empty()) {
      std::cerr << "video does not contain enough background frames"
                << std::endl;
      return true;
    }

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

void find_particles(cv::InputArray &frame, cv::InputArray &mean,
                    cv::InputArray &var, const double zscore,
                    cv::InputArray &mask, const double unsharp_alpha,
                    std::vector<Particle> &particles, const int current_frame,
                    int current_id) {
  // calculate the difference between frame and mean
  cv::UMat diff;
  frame.copyTo(diff);
  diff.convertTo(diff, CV_32F);
  // frame.getMat().convertTo(diff, CV_32F);
  cv::subtract(diff, mean, diff);
  cv::multiply(diff, -1.f, diff);

  // median blur
  cv::medianBlur(diff, diff, 3);

  // sharpen
  if (unsharp_alpha > 0.0)
    unsharp_mask(diff, diff, unsharp_alpha);

  // mask differences below x std deviations
  cv::UMat std;
  cv::sqrt(var, std);
  cv::multiply(std, zscore, std);

  cv::UMat thresh = cv::UMat(frame.rows(), frame.cols(), CV_8U);
  cv::compare(diff, std, thresh, cv::CMP_GT);

  cv::bitwise_and(thresh, mask, thresh);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresh, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  cv::Mat cpu_diff = diff.getMat(cv::ACCESS_READ);
  particles.reserve(contours.size());
  std::transform(
      contours.begin(), contours.end(), std::back_inserter(particles),
      [&](const std::vector<cv::Point> &contour) {
        return Particle(contour, cpu_diff, current_frame, current_id++);
      });
}
