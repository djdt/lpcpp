#pragma once

#include <opencv2/core.hpp>
#include <vector>

class Particle {
private:
  std::vector<cv::Point> _contour;
  cv::Mat _image;
  cv::Mat _image_raw;
  cv::Mat _mask;
  cv::Rect _rect;
  cv::RotatedRect _min_area_rect;

  static long id_counter;

  long _id;
  int _frame;
  int _frame_count;

  cv::Moments _moments;

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
  double min_area = 5.0;
  double max_area = 1e4;
  double min_aspect = 0.5;
  double max_aspect = 1.0;
  double min_circularity = 0.5;
  double max_circularity = 1.0;
  double min_convexity = 0.5;
  double max_convexity = 1.0;
  double min_radius = 1.0;
  double max_radius = 1e3;
  double min_intensity = 1e3;
  double max_intensity = 1e6;
  double min_sharpness = 0.0;
  double max_sharpness = 0.0;
};
