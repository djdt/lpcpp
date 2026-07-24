#pragma once

#include <opencv2/core/types.hpp>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

enum ParticleFrameMetric {
  METRIC_AVERAGE_INTENSITY,
  METRIC_CENTER_WEIGHTED_INTENSITY,
  METRIC_SHARPNESS,
};

class Particle {
private:
  static long id_counter;
  long _id;

  ParticleFrameMetric _metric_method;
  double _metric;

  std::vector<int> _frames;
  std::vector<cv::Mat> _images;
  std::vector<cv::Mat> _raw_images;
  std::vector<std::vector<cv::Point>> _contours;
  // std::vector<cv::Moments> _contour_moments;

  cv::KalmanFilter _kalman; // position tracking

  size_t _index;

public:
  // ensure a cv::Mat here
  Particle(const int frame_number, const std::vector<cv::Point> &contour,
           const cv::Mat &image, const cv::Mat &raw_image,
           ParticleFrameMetric metric = METRIC_CENTER_WEIGHTED_INTENSITY);

  const int frameCount() const;
  const long id() const;

  const int lastFrame() const;

  // current index access
  const std::vector<cv::Point> &contour(const int index = -1) const;
  const int frame(const int index = -1) const;
  const cv::Mat &image(const int index = -1) const;
  const cv::Moments &moments(const int index = -1) const;
  const cv::Mat &rawImage(const int index = -1) const;

  const cv::Rect boundingRect() const;
  void update(const int frame_number, const std::vector<cv::Point> &contour,
              const cv::Mat &image, const cv::Mat &raw_image);

  cv::Point2f position() const;
  cv::Point2f velocity() const;
  cv::Point2f predictedPosition(const int frame) const;
};

double calculate_selection_metric(const std::vector<cv::Point> &contour,
                                  cv::InputArray &image,
                                  ParticleFrameMetric metric);
