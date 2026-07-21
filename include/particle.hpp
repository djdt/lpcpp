#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>

enum ParticleMetric {
  PM_AVERAGE_DARK,
  PM_AVERAGE_LIGHT,
  PM_AVERAGE_ABS,
  PM_CENTER_WEIGHTED_DARK,
  PM_CENTER_WEIGHTED_LIGHT,
  PM_CENTER_WEIGHTED_ABS,
  PM_SHARPNESS,
};

class Particle {
private:
  static long id_counter;
  long _id;

  ParticleMetric _metric_method;
  double _metric;

  std::vector<cv::Mat> _images;
  std::vector<cv::Mat> _raw_images;
  std::vector<std::vector<cv::Point>> _contours;
  std::vector<int> _frames;

  size_t _index;

public:
  // ensure a cv::Mat here
  Particle(const int frame_number, const std::vector<cv::Point> &contour,
           const cv::Mat &image, const cv::Mat &raw_image,
           ParticleMetric metric = PM_CENTER_WEIGHTED_DARK);

  const int frameCount() const;
  const long id() const;

  const int lastFrame() const;

  // current index access
  const int frame(const int index = -1) const;
  const std::vector<cv::Point> &contour(const int index = -1) const;
  const cv::Mat &image(const int index = -1) const;
  const cv::Mat &rawImage(const int index = -1) const;

  const cv::Rect boundingRect() const;
  void update(const int frame_number, const std::vector<cv::Point> &contour,
              const cv::Mat &image, const cv::Mat &raw_image);
};

double calculate_selection_metric(const std::vector<cv::Point> &contour,
                                  cv::InputArray &image, ParticleMetric metric);
