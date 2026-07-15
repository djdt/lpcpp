#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

struct filter_args {
  std::pair<double, double> area = {5.0, 1e4};
  std::pair<double, double> aspect = {0.5, 1.0};
  std::pair<double, double> circularity = {0.5, 1.0};
  std::pair<double, double> convexity = {0.5, 1.0};
  std::pair<double, double> intensity = {1e3, 1e6};
  std::pair<double, double> radius = {1.0, 11e3};
  std::pair<double, double> sharpness = {0.0, 0.0};
};

// double contour_area(const std::vector<cv::Point> &contour);
double contour_aspect(const std::vector<cv::Point> &contour);

double contour_weighted_intensity(const std::vector<cv::Point> &contour,
                                  cv::InputArray &image,
                                  const cv::Point &offset = cv::Point());
double
contour_circular_equivalent_diameter(const std::vector<cv::Point> &contour,
                                     const double area = -1.0);
double contour_circularity(const std::vector<cv::Point> &contour,
                           const double area = -1.0);
double contour_convexity(const std::vector<cv::Point> &contour,
                         const double area = -1.0);
double contour_distance(const std::vector<cv::Point> &contour,
                        const cv::Point2f &pos);
double contour_distance(const std::vector<cv::Point> &contour,
                        const std::vector<cv::Point> &contour2);
double contour_edge_distance(const std::vector<cv::Point> &contour,
                             const cv::Point2f &pos);
double contour_intensity(const std::vector<cv::Point> &contour,
                         cv::InputArray &image,
                         const cv::Point &offset = cv::Point(0, 0));
double contour_maximum_feret(const std::vector<cv::Point> &contour);
double contour_minimum_feret(const std::vector<cv::Point> &contour);

void filter_contours(std::vector<std::vector<cv::Point>> &contours,
                     const cv::Mat &frame, struct filter_args args);

void mask_for_contour(const std::vector<cv::Point> &contour,
                      cv::InputOutputArray &mask);
