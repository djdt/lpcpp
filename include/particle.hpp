#pragma once

#include <opencv2/core/types.hpp>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>

typedef std::vector<cv::Point> Contour;

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
  std::vector<std::pair<Contour, cv::Moments>> _contours;

  cv::KalmanFilter _kalman; // position tracking

  size_t _index;

public:
  // ensure a cv::Mat here
  Particle(const int frame_number,
           const std::pair<Contour, cv::Moments> &contour_pair,
           const cv::Mat &image, const cv::Mat &raw_image,
           ParticleFrameMetric metric = METRIC_CENTER_WEIGHTED_INTENSITY);

  void initTrajectory(); // separated for performance

  const cv::Rect boundingRect() const;

  const int frameCount() const;
  const long id() const;

  const int lastFrame() const;
  const Contour &lastContour() const;

  // current index access
  const Contour &contour(const int index = -1) const;
  const int frame(const int index = -1) const;
  const cv::Mat &image(const int index = -1) const;
  const cv::Moments &moments(const int index = -1) const;
  const cv::Mat &rawImage(const int index = -1) const;
  void update(const int frame_number,
              const std::pair<Contour, cv::Moments> &contour_pair,
              const cv::Mat &image, const cv::Mat &raw_image);
  void updateTrajectory();

  cv::Point2f position() const;
  cv::Point2f velocity() const;
  cv::Point2f predictedPosition(const int frame) const;
  std::vector<cv::Point> trajectory(const int frame_count) const;
};

double
calculate_selection_metric(const std::pair<Contour, cv::Moments> &contour_pair,
                           cv::InputArray &image, ParticleFrameMetric metric);
