#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

enum ParticleMetric { CENTER_WEIGHTED_INTENSITY = 0, INTENSITY = 1, SHARPNESS };

class Particle {
private:
  static long id_counter;
  const long _id;

  const ParticleMetric _metric_method;
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
           ParticleMetric metric = CENTER_WEIGHTED_INTENSITY);

  void update(const int frame_number, const std::vector<cv::Point> &contour,
              const cv::Mat &image, const cv::Mat &raw_image);

  const int frameCount() const;
  const long id() const;

  // current index access
  const int frame() const;
  const std::vector<cv::Point> &contour() const;
  const cv::Mat &image() const;
  const cv::Mat &rawImage() const;
};

double calculate_selection_metric(const std::vector<cv::Point> &contour,
                                  cv::InputArray &image, ParticleMetric metric);
