#pragma once

#include <opencv2/core.hpp>
#include <vector>

class Particle {
private:
  cv::Mat _image;
  cv::Mat _image_raw;
  cv::Mat _mask;
  cv::Rect _rect;
  cv::RotatedRect _min_area_rect;

  static long id_counter;

  long _id;
  int _frame_count;

  cv::Moments _moments;

  std::vector<std::vector<cv::Point>> _contours;
  std::vector<int> _frames;

public:
  // ensure a cv::Mat here
  Particle(const std::vector<cv::Point> &contour, const cv::Mat &frame,
           int frame_number);

  const std::vector<cv::Point> &contour() const;
  const int frameNumber() const;
  int frameCount() const;
  const long id() const;
  const cv::Mat &image() const;
  const cv::Mat &rawImage() const;

  const std::vector<cv::Point> imageContour() const;

  double area() const;
  double aspect() const;
  cv::Point2f center() const;
  const double centerWeightedIntensity() const;
  const double circularEquvalentDiameter() const;
  double circularity() const;
  double convexity() const;
  double intensity() const;
  const double maximumWidth() const;
  const double minimumWidth() const;
  const double perimeter() const;
  double radius() const;
  double sharpness() const;

  void addFrames(const int count);
  void setRawImage(const cv::Mat &frame);
  bool isClose(const Particle &b, const double edge_distance = 0.0);
};

struct filter_args {
  std::pair<double, double> area = {5.0, 1e4};
  std::pair<double, double> aspect = {0.5, 1.0};
  std::pair<double, double> circularity = {0.5, 1.0};
  std::pair<double, double> convexity = {0.5, 1.0};
  std::pair<double, double> intensity = {1e3, 1e6};
  std::pair<double, double> radius = {1.0, 11e3};
  std::pair<double, double> sharpness = {0.0, 0.0};
};
