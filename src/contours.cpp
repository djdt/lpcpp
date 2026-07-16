#include "contours.hpp"
#include "cpuproc.hpp"

#include <algorithm>
#include <cmath>
#include <execution>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

double contour_aspect(const std::vector<cv::Point> &contour) {
  cv::RotatedRect rect = cv::minAreaRect(contour);
  double aspect = rect.size.aspectRatio();
  if (aspect > 1.0) {
    aspect = 1.0 / aspect;
  }
  return aspect;
}

double
contour_circular_equivalent_diameter(const std::vector<cv::Point> &contour,
                                     const double area) {
  if (area < 0.0)
    double area = cv::contourArea(contour);
  return std::sqrt((4.0 * area) / std::numbers::pi);
}

double contour_circularity(const std::vector<cv::Point> &contour,
                           const double area) {
  if (area < 0.0)
    double area = cv::contourArea(contour);
  auto perim = cv::arcLength(contour, true);
  return std::sqrt(4.0 * std::numbers::pi * area / std::pow(perim, 2));
}

double contour_convexity(const std::vector<cv::Point> &contour,
                         const double area) {
  std::vector<cv::Point> hull;
  cv::convexHull(contour, hull);
  return area / cv::contourArea(hull);
}

double contour_distance(const std::vector<cv::Point> &contour,
                        const cv::Point2f &pos) {
  // double min_dist = std::numeric_limits<double>::infinity();
  // for (const cv::Point2f &c : contour) {
  //   double dist = cv::norm(c - pos);
  //   if (dist < min_dist) {
  //     min_dist = dist;
  //   }
  // }
  // return min_dist * -cv::pointPolygonTest(contour, pos, false);
  double dist = cv::pointPolygonTest(contour, pos, true);
  return -dist;
}

double contour_distance(const std::vector<cv::Point> &contour_a,
                        const std::vector<cv::Point> &contour_b) {
  // possible to approximate with getClosestEllipsePoints
  std::vector<double> dists;
  dists.reserve(contour_a.size());
  std::transform(contour_a.begin(), contour_a.end(), std::back_inserter(dists),
                 [&contour_b](const cv::Point &p) {
                   return contour_distance(contour_b, p);
                 });
  return *std::min_element(dists.begin(), dists.end());
}

double contour_maximum_feret(const std::vector<cv::Point> &contour) {
  auto rect = cv::minAreaRect(contour);
  return std::max(rect.size.width, rect.size.height);
}

double contour_minimum_feret(const std::vector<cv::Point> &contour) {
  auto rect = cv::minAreaRect(contour);
  return std::min(rect.size.width, rect.size.height);
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
              double aspect = contour_aspect(c);
              if (aspect < args.aspect.first or aspect > args.aspect.second) {
                return true;
              }
            }
            if (args.circularity.first != args.circularity.second) {
              double circularity = contour_circularity(c, moments.m00);
              if (circularity < args.circularity.first or
                  circularity > args.circularity.second) {
                return true;
              }
            }
            if (args.convexity.first != args.convexity.second) {
              double convexity = contour_convexity(c, moments.m00);
              if (convexity < args.convexity.first or
                  convexity > args.convexity.second) {
                return true;
              }
            }
            if (args.radius.first != args.radius.second) {
              double radius =
                  contour_distance(c, cv::Point2f(moments.m00 / moments.m10,
                                                  moments.m00 / moments.m01));
              if (radius < args.radius.first or radius > args.radius.second) {
                return true;
              }
            }
            // image based operations
            if ((args.intensity.first != args.intensity.second) ||
                (args.sharpness.first != args.sharpness.second)) {
              cv::Mat mask;
              cv::Rect rect = cv::boundingRect(c);
              mask_for_contour(c, mask);

              if (args.intensity.first != args.intensity.second) {
                double intensity = image_intensity(frame(rect), mask);
                if (intensity < args.intensity.first or
                    intensity > args.intensity.second) {
                  return true;
                }
              }
              if (args.sharpness.first != args.sharpness.second) {
                cv::Mat laplace;
                double sharpness = image_sharpness(frame(rect), laplace);
                if (sharpness < args.sharpness.first or
                    sharpness > args.sharpness.second) {
                  return true;
                }
              }
            }
            return false;
          }),
      contours.end());
}

void mask_for_contour(const std::vector<cv::Point> &contour,
                      cv::InputOutputArray &mask) {
  cv::Rect rect = cv::boundingRect(contour);
  mask.create(rect.size(), CV_8U);
  cv::drawContours(mask, {contour}, 0, 255, -1, cv::LINE_8, cv::noArray(), 0,
                   -rect.tl());
}
