#include "cpuproc.hpp"

#include "util.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgproc.hpp>

std::array<float, 3> find_capillary(cv::InputArray &input) {
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(input, circles, cv::HOUGH_GRADIENT, 1.0,
                   static_cast<float>(input.rows()) / 2.f, 50, 5,
                   input.rows() / 4, input.rows());

  if (circles.size() == 0) {
    return {0.f, 0.f, 0.f};
  }
  return {circles[0][0], circles[0][1], circles[0][2]};
}

double image_center_weighted_intensity(cv::InputArray &image,
                                       cv::InputArray &mask,
                                       cv::OutputArray &weights) {
  weights.createSameSize(image, CV_32F);
  cv::distanceTransform(mask, weights, cv::DIST_L2, cv::DIST_MASK_3);
  cv::multiply(weights, image, weights, 1.0 / cv::sum(weights)[0]);
  return cv::sum(weights)[0];
}

double image_intensity(cv::InputArray &image, cv::InputArray &mask) {
  if (mask.empty()) {
    return cv::sum(image)[0];
  }
  return cv::mean(image, mask)[0] * cv::countNonZero(mask);
}

double image_sharpness(cv::InputArray &image, cv::OutputArray &laplace) {
  laplace.createSameSize(image, CV_32F);

  cv::Laplacian(image, laplace, CV_32F);
  cv::Scalar mu, sigma;
  cv::meanStdDev(laplace, mu, sigma);
  return sigma[0];
}

void unsharp_mask(cv::InputArray &image, cv::OutputArray &output,
                  double alpha = 1.0) {
  output.createSameSize(image, CV_32F);

  cv::UMat sobelx, sobely, mag;
  cv::Sobel(image, sobelx, CV_32F, 1, 0, 3);
  cv::Sobel(image, sobely, CV_32F, 0, 1, 3);
  cv::magnitude(sobelx, sobely, mag);
  cv::addWeighted(image, 1.0 + alpha, mag, -alpha, 0, output);
}

void update_background(cv::InputArray &frame, cv::InputOutputArray &mean,
                       cv::InputOutputArray &var, int pos) {
  double weight = 1.0 / std::max(1.0, static_cast<double>(pos));

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

void preprocess_and_threshold(cv::InputArray &frame, cv::InputArray &mean,
                              cv::InputArray &var, cv::OutputArray &processed,
                              cv::OutputArray &threshold, const double zscore,
                              const double unsharp_alpha,
                              const PreprocessImageMode mode) {
  processed.createSameSize(frame, CV_32F);
  threshold.createSameSize(frame, CV_8U);

  switch (mode) {
  case PROC_MODE_ABSOLUTE: {
    cv::absdiff(frame, mean, processed);
    break;
  }
  case PROC_MODE_NORMAL: {
    cv::subtract(frame, mean, processed, cv::noArray(), CV_32F);
    break;
  }
  case PROC_MODE_INVERT: {
    // same as normal * -1
    cv::subtract(mean, frame, processed, cv::noArray(), CV_32F);
    break;
  }
  default:
    throw "unknown processing method ";
  }

  // sharpen image to reduce particle edge blur
  if (unsharp_alpha > 0.0)
    unsharp_mask(processed, processed, unsharp_alpha);

  // mask differences below x std deviations
  cv::UMat std;
  cv::sqrt(var, std);
  cv::multiply(std, zscore, std);

  cv::compare(processed, std, threshold, cv::CMP_GT);
}
