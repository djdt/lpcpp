#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "asynccapture.hpp"
#include "particle.hpp"
#include "util.hpp"

bool mask_capillary(cv::InputArray &input, cv::Mat &mask, double &um_per_px,
                    const double capillary_diameter = 750.0) {
  cv::Mat frame = input.getMat();

  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(frame, circles, cv::HOUGH_GRADIENT, 1.0,
                   static_cast<float>(frame.rows) / 2.f, 50, 5, frame.rows / 4,
                   frame.rows);

  if (circles.size() == 0) {
    std::cerr << "\tcould not detect capillary" << std::endl;
    return true;
  } else {
    std::cout << "\tcapillary detected at " << circles[0][0] << " x "
              << circles[0][1] << " with radius " << circles[0][2] << std::endl;
  }

  um_per_px = capillary_diameter / (2.0 * circles[0][2]);
  mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
  cv::circle(mask, cv::Point(circles[0][0], circles[0][1]), circles[0][2] * 0.9,
             255, -1);
  return false;
}

void unsharp_mask(const cv::Mat &image, cv::Mat &output, double alpha = 1.0) {
  cv::Mat sobelx, sobely, mag;
  cv::Sobel(image, sobelx, 1, 0, 3);
  cv::Sobel(image, sobely, 0, 1, 3);
  cv::magnitude(sobelx, sobely, mag);
  cv::addWeighted(image, 1.0 + alpha, mag, -alpha, 0, output);
}

void update_background(const cv::Mat &frame, cv::Mat &mean, cv::Mat &var,
                       int pos) {
  double weight = 1.0 / static_cast<double>(pos);

  cv::Mat frame_var;
  frame.convertTo(frame_var, CV_32F);

  cv::addWeighted(frame_var, weight, mean, 1.0 - weight, 0.0, mean);

  cv::subtract(frame_var, mean, frame_var);
  cv::pow(frame_var, 2.0, frame_var);

  cv::addWeighted(frame_var, weight, var, 1.0 - weight, 0.0, var);
}

bool init_background(AsyncVideoCapture &cap, cv::Mat &mean, cv::Mat &var,
                     int frame_count) {
  int frame_pos = 0;
  cap.set(cv::CAP_PROP_POS_FRAMES, frame_pos);
  cap.invalidate();

  cv::Mat frame;

  auto start_time = std::chrono::system_clock::now();

  while (frame_pos++ < frame_count) {
    cap.read(frame);
    if (frame.empty()) {
      std::cerr << "video does not contain enough background frames"
                << std::endl;
      return true;
    }
    cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

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

void find_particles(const cv::Mat &frame, const cv::Mat &mean,
                    const cv::Mat &var, const double zscore,
                    const cv::Mat &mask, std::vector<Particle> &particles,
                    const int current_frame, int current_id) {

  // calculate the difference between frame and mean
  cv::Mat diff;
  frame.convertTo(diff, CV_32F);
  cv::subtract(diff, mean, diff);
  diff *= -1.f;

  // median blur
  cv::medianBlur(diff, diff, 3);

  // sharpen
  unsharp_mask(diff, diff, 1.0);

  // mask differences below x std deviations
  cv::Mat std;
  cv::sqrt(var, std);
  std *= zscore;

  cv::Mat thresh = cv::Mat(frame.rows, frame.cols, CV_8U);
  cv::compare(diff, std, thresh, cv::CMP_GT);

  cv::bitwise_and(diff > zscore * std, mask, thresh);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresh, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  particles.reserve(contours.size());
  std::transform(contours.begin(), contours.end(),
                 std::back_inserter(particles),
                 [&](const std::vector<cv::Point> &contour) {
                   return Particle(contour, diff, current_frame, current_id++);
                 });
}
