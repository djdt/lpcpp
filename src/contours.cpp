#include "contours.hpp"
#include "cpuproc.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

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

double contour_box_distance(const std::vector<cv::Point> &contour_a,
                            const std::vector<cv::Point> &contour_b) {
  cv::Rect rect_a = cv::boundingRect(contour_a);
  cv::Rect rect_b = cv::boundingRect(contour_b);
  cv::Point2f center_a = cv::Point2f(rect_a.x + rect_a.width / 2.f,
                                     rect_a.y + rect_a.height / 2.f);
  cv::Point2f center_b = cv::Point2f(rect_b.x + rect_b.width / 2.f,
                                     rect_b.y + rect_b.height / 2.f);
  return cv::norm(center_a - center_b) - (cv::norm(rect_a.tl() - rect_a.br()) +
                                          cv::norm(rect_b.tl() - rect_b.br())) /
                                             2.0;
}

double contour_circle_distance(const std::vector<cv::Point> &contour_a,
                               const std::vector<cv::Point> &contour_b) {
  cv::Point2f center_a, center_b;
  float radius_a, radius_b;
  cv::minEnclosingCircle(contour_a, center_a, radius_a);
  cv::minEnclosingCircle(contour_b, center_b, radius_b);
  return cv::norm(center_a - center_b) - (radius_a + radius_b);
}
double contour_distance(const std::vector<cv::Point> &contour,
                        const cv::Point2f &pos) {
  return -cv::pointPolygonTest(contour, pos, true);
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
                     const cv::Mat &frame, const filter_args &args) {
  auto it = std::remove_if(
      contours.begin(), contours.end(), [=](const std::vector<cv::Point> &c) {
        cv::Moments moments = cv::moments(c);
        if (args.area.first != args.area.second) {
          if (moments.m00 < args.area.first || moments.m00 > args.area.second) {
            return true;
          }
        }
        if (args.aspect.first != args.aspect.second) {
          double aspect = contour_aspect(c);
          if (aspect < args.aspect.first || aspect > args.aspect.second) {
            return true;
          }
        }
        if (args.circularity.first != args.circularity.second) {
          double circularity = contour_circularity(c, moments.m00);
          if (circularity < args.circularity.first ||
              circularity > args.circularity.second) {
            return true;
          }
        }
        if (args.convexity.first != args.convexity.second) {
          double convexity = contour_convexity(c, moments.m00);
          if (convexity < args.convexity.first ||
              convexity > args.convexity.second) {
            return true;
          }
        }
        if (args.radius.first != args.radius.second) {
          double radius =
              contour_distance(c, cv::Point2f(moments.m00 / moments.m10,
                                              moments.m00 / moments.m01));
          if (radius < args.radius.first || radius > args.radius.second) {
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
            if (intensity < args.intensity.first ||
                intensity > args.intensity.second) {
              return true;
            }
          }
          if (args.sharpness.first != args.sharpness.second) {
            cv::Mat laplace;
            double sharpness = image_sharpness(frame(rect), laplace);
            if (sharpness < args.sharpness.first ||
                sharpness > args.sharpness.second) {
              return true;
            }
          }
        }
        return false;
      });
  contours.erase(it, contours.end());
}

void mask_for_contour(const std::vector<cv::Point> &contour,
                      cv::InputOutputArray &mask) {
  cv::Rect rect = cv::boundingRect(contour);
  mask.create(rect.size(), CV_8U);
  mask.setTo(0);
  cv::drawContours(mask, {contour}, 0, 255, -1, cv::LINE_8, cv::noArray(), 0,
                   -rect.tl());
}
