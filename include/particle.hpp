#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

enum particle_metric { CENTER_WEIGHTED_INTENSITY, INTENSITY, SHARPNESS };

class Particle {
private:
  static long id_counter;
  long _id;
  double _metric;
  size_t _index;

  std::vector<cv::Mat> _images;
  std::vector<cv::Mat> _raw_images;
  std::vector<std::vector<cv::Point>> _contours;
  std::vector<int> _frames;

public:
  // ensure a cv::Mat here
  Particle(int frame_number, const std::vector<cv::Point> &contour,
           const cv::Mat &image, const cv::Mat &raw_image,
           particle_metric metric = CENTER_WEIGHTED_INTENSITY);

  void update(int frame_number, const std::vector<cv::Point> &contour,
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
                                  cv::InputArray &image,
                                  particle_metric metric);
