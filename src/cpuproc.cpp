#include <algorithm>
#include <execution>
#include <iomanip>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <vector>

#include "particle.hpp"
#include "util.hpp"

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

void difference_from_mean(cv::InputArray &frame, cv::InputArray &mean,
                          cv::OutputArray &diff) {
  cv::subtract(frame, mean, diff, cv::noArray(), CV_32F);
  cv::multiply(diff, -1.f, diff);
}

void niblack_threshold(cv::InputArray &image, cv::InputArray &mean,
                       cv::InputArray &var, cv::OutputArray &threshold,
                       const double zscore = 3.0) {
  // mask differences below x std deviations
  cv::UMat std;
  cv::sqrt(var, std);
  cv::multiply(std, zscore, std);

  threshold.create(image.size(), CV_8U);
  cv::compare(image, std, threshold, cv::CMP_GT);
}

// void find_contours(cv::InputOutputArray &diff, cv::InputArray &mean,
//                    cv::InputArray &var, const double zscore,
//                    cv::InputArray &mask, const double unsharp_alpha,
//                    std::vector<std::vector<cv::Point>> &contours) {
//
//   cv::subtract(diff, mean, diff);
//   cv::multiply(diff, -1.f, diff);
//
//   // median blur
//   cv::medianBlur(diff, diff, 5);
//
//   // sharpen
//   if (unsharp_alpha > 0.0)
//     unsharp_mask(diff, diff, unsharp_alpha);
//
//   // mask differences below x std deviations
//   cv::UMat std;
//   cv::sqrt(var, std);
//   cv::multiply(std, zscore, std);
//
//   cv::UMat thresh = cv::UMat(diff.rows(), diff.cols(), CV_8U);
//   cv::compare(diff, std, thresh, cv::CMP_GT);
//
//   cv::bitwise_and(thresh, mask, thresh);
//
//   cv::findContours(thresh, contours, cv::RETR_EXTERNAL,
//                    cv::CHAIN_APPROX_SIMPLE);
//
//   // filter_contours(contours, cpu_diff);
//   // const cv::Mat cpu_diff = diff.getMat(cv::ACCESS_READ);
//   // particles.reserve(contours.size());
//   // std::transform(contours.begin(), contours.end(),
//   //                std::back_inserter(particles),
//   //                [&](const std::vector<cv::Point> &contour) {
//   //                  return Particle(contour, cpu_diff, current_frame);
//   //                });
// }

double contour_mean_radius(const std::vector<cv::Point> contour,
                           const cv::Moments &moments) {
  const cv::Point2f c(moments.m10 / moments.m00, moments.m01 / moments.m00);
  double dist = std::accumulate(
      contour.begin(), contour.end(), 0.0,
      [&c](double sum, const cv::Point2f &p) { return sum + cv::norm(p - c); });
  return dist / contour.size();
}

double image_sharpness(const cv::Mat &image) {
  cv::Mat laplace;
  cv::Laplacian(image, laplace, CV_32F);
  cv::Scalar mu, sigma;
  cv::meanStdDev(laplace, mu, sigma);
  return sigma[0];
}

double center_weighted_intensity(const std::vector<cv::Point> &contour,
                                 cv::InputArray &image) {
  cv::Rect rect = cv::boundingRect(contour);
  cv::Mat mask = cv::Mat::zeros(rect.size(), CV_8U);
  cv::drawContours(mask, {contour}, 0, 255, -1, cv::LINE_8, cv::noArray(), 0,
                   -rect.tl());
  cv::Mat weights(rect.size(), CV_32F);
  cv::distanceTransform(mask, weights, cv::DIST_L2, cv::DIST_MASK_3);
  int kind = image.kind();

  if (image.isMat()) {
    cv::multiply(weights, image.getMat()(rect), weights,
                 1.0 / cv::sum(weights)[0]);
  } else if (image.isUMat()) {
    cv::multiply(weights, image.getUMat()(rect), weights,
                 1.0 / cv::sum(weights)[0]);
  } else {
    throw "only Mat and UMat are supported";
  }

  return cv::sum(weights)[0];
}

void filter_contours(std::vector<std::vector<cv::Point>> &contours,
                     const cv::Mat &frame, struct filter_args args) {
  contours.erase(
      std::remove_if(
          std::execution::par, contours.begin(), contours.end(),
          [=](const std::vector<cv::Point> &c) {
            cv::Moments moments = cv::moments(c);
            if (args.area.first != args.area.second) {
              if (moments.m00 < args.area.first or
                  moments.m00 > args.area.second) {
                return true;
              }
            }
            if (args.aspect.first != args.aspect.second) {
              auto rect = cv::minAreaRect(c);
              double aspect = rect.size.aspectRatio();
              if (aspect > 1.0)
                aspect = 1.0 / aspect;
              if (aspect < args.aspect.first or aspect > args.aspect.second) {
                return true;
              }
            }
            if (args.circularity.first != args.circularity.second) {
              double perim = cv::arcLength(c, true);
              double circularity =
                  4.0 * std::numbers::pi * moments.m00 / std::pow(perim, 2);
              if (circularity < args.circularity.first or
                  circularity > args.circularity.second) {
                return true;
              }
            }
            if (args.convexity.first != args.convexity.second) {
              std::vector<cv::Point> hull;
              cv::convexHull(c, hull);
              double convexity = moments.m00 / cv::contourArea(hull);
              if (convexity < args.convexity.first or
                  convexity > args.convexity.second) {
                return true;
              }
            }
            if (args.radius.first != args.radius.second) {
              double radius = contour_mean_radius(c, moments);
              if (radius < args.radius.first or radius > args.radius.second) {
                return true;
              }
            }
            if (args.intensity.first != args.intensity.second) {
              cv::Rect rect = cv::boundingRect(c);
              cv::Mat mask = cv::Mat::zeros(rect.size(), CV_8U);
              cv::Mat image = cv::Mat::zeros(rect.size(), CV_8U);
              cv::drawContours(mask, {c}, 0, 255, -1, cv::LINE_8, cv::noArray(),
                               0, -rect.tl());

              frame(rect).copyTo(image, mask);
              double intensity = cv::sum(intensity)[0];
              if (intensity < args.intensity.first or
                  intensity > args.intensity.second) {
                return true;
              }
            }
            if (args.sharpness.first != args.sharpness.second) {
              cv::Rect rect = cv::boundingRect(c);
              double sharpness = image_sharpness(frame(rect));
              if (sharpness < args.sharpness.first or
                  sharpness > args.sharpness.second) {
                return true;
              }
            }
            return false;
          }),
      contours.end());
}

void filter_existing_particles(
    std::vector<Particle> &old_particles, std::vector<Particle> &new_particles,
    const std::function<bool(const Particle &, const Particle &)> comparison,
    const double edge_distance) {
  std::vector<size_t> remove_new_at;
  old_particles.erase(
      std::remove_if(
          std::execution::seq, old_particles.begin(), old_particles.end(),
          [&](Particle &old) {
            for (auto it_new = new_particles.begin();
                 it_new != new_particles.end(); ++it_new) {

              if (it_new->isClose(old, edge_distance)) {

                if (comparison(*it_new, old)) {
                  it_new->addFrames(
                      old.frameCount()); // inherit old particle count
                  return true;           // old is removed
                } else {
                  // remove new
                  size_t idx = std::distance(new_particles.begin(), it_new);
                  if (remove_new_at.size() == 0 or
                      remove_new_at.back() != idx) {
                    remove_new_at.push_back(idx);
                  }
                  old.addFrames(1);
                  return false;
                }
              }
            }
            return false;
          }),
      old_particles.end());

  // sort and remove non-unqiue indicies
  std::sort(remove_new_at.begin(), remove_new_at.end());
  auto last = std::unique(remove_new_at.begin(), remove_new_at.end());
  remove_new_at.erase(last, remove_new_at.end());

  new_particles.erase(remove_indices(new_particles.begin(), new_particles.end(),
                                     remove_new_at.begin(),
                                     remove_new_at.end()),
                      new_particles.end());
}
