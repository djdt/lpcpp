#pragma once

#include <vector>

#include <opencv2/core.hpp>

typedef std::vector<cv::Point> Contour;

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

// double contour_area(const Contour &contour);
double contour_aspect(const Contour &contour);

// cv::Point2f contour_center(const Contour &contour);

double contour_circular_equivalent_diameter(const Contour &contour,
                                            const double area);

double contour_circularity(const Contour &contour, const double area);
double contour_convexity(const Contour &contour, const double area);

double contour_edge_distance_box(const Contour &contour_a,
                                 const Contour &contour_b);

double contour_edge_distance_circle(const Contour &contour_a,
                                    const Contour &contour_b);

double contour_edge_distance(const Contour &contour, const Contour &contour2);
double contour_edge_distance(const Contour &contour, const cv::Point2f &pos);

// double contour_mean_diameter(const Contour &contour);
double contour_mean_distance(const Contour &contour, const cv::Point2f &pos);

double contour_maximum_feret(const Contour &contour);
double contour_minimum_feret(const Contour &contour);

void filter_contours(std::vector<std::pair<Contour, cv::Moments>> &contours,
                     const cv::UMat &frame, const filter_args &args);

void mask_for_contour(const Contour &contour, cv::InputOutputArray &mask);

cv::Point2f legendre_axes_from_moments(const cv::Moments &moments);
