#pragma once

#include <limits>
#include <vector>

#include <opencv2/core.hpp>

struct filter_args {
  std::pair<double, double> area = {5.0, 1e4};
  std::pair<double, double> aspect = {0.5, 1.0};
  std::pair<double, double> circularity = {0.5, 1.0};
  std::pair<double, double> convexity = {0.5, 1.0};
  std::pair<double, double> intensity = {1e3, 1e6};
  std::pair<double, double> radius = {1.0, 11e3};
  std::pair<double, double> sharpness = {0.0, 0.0};
};

double box_edge_distance(const cv::Rect &rect_a, const cv::Rect &rect_b);

// double contour_area(const std::vector<cv::Point> &contour);
double contour_aspect(const std::vector<cv::Point> &contour);

cv::Point2f contour_center(const std::vector<cv::Point> &contour);

double contour_circular_equivalent_diameter(
    const std::vector<cv::Point> &contour,
    const double area = std::numeric_limits<double>::quiet_NaN());

double contour_circularity(
    const std::vector<cv::Point> &contour,
    const double area = std::numeric_limits<double>::quiet_NaN());
double
contour_convexity(const std::vector<cv::Point> &contour,
                  const double area = std::numeric_limits<double>::quiet_NaN());

double contour_edge_distance_box(const std::vector<cv::Point> &contour_a,
                                 const std::vector<cv::Point> &contour_b);

double contour_edge_distance_circle(const std::vector<cv::Point> &contour_a,
                                    const std::vector<cv::Point> &contour_b);

double contour_edge_distance(const std::vector<cv::Point> &contour,
                             const std::vector<cv::Point> &contour2);
double contour_edge_distance(const std::vector<cv::Point> &contour,
                             const cv::Point2f &pos);

double contour_mean_diameter(const std::vector<cv::Point> &contour);
double contour_mean_distance(const std::vector<cv::Point> &contour,
                             const cv::Point2f &pos);

double contour_maximum_feret(const std::vector<cv::Point> &contour);
double contour_minimum_feret(const std::vector<cv::Point> &contour);

void filter_contours(std::vector<std::vector<cv::Point>> &contours,
                     const cv::Mat &frame, const filter_args &args);

void mask_for_contour(const std::vector<cv::Point> &contour,
                      cv::InputOutputArray &mask);

cv::Point2f legendre_axes_from_moments(const cv::Moments &moments);
