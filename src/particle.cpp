#include "particle.hpp"

#include "contours.hpp"
#include "cpuproc.hpp"

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>

Particle::Particle(const int frame_number,
                   const std::vector<cv::Point> &contour, const cv::Mat &image,
                   const cv::Mat &raw_image, const ParticleFrameMetric method)
    : _id(id_counter++), _index(0), _metric_method(method) {

  cv::Rect rect = cv::boundingRect(contour);
  rect &= cv::Rect(0, 0, image.cols, image.rows);

  _frames.push_back(frame_number);
  _contours.push_back(contour);
  _images.push_back(image(rect).clone());
  if (!raw_image.empty()) {
    _raw_images.push_back(raw_image(rect).clone());
  }

  _metric = calculate_selection_metric(contour, _images.back(), _metric_method);
};

const int Particle::frameCount() const { return _frames.size(); }
const long Particle::id() const { return _id; }

const int Particle::lastFrame() const { return _frames.back(); }

const int Particle::frame(const int index) const {
  if (index < 0)
    return _frames[_index];
  return _frames[index];
}
const std::vector<cv::Point> &Particle::contour(const int index) const {
  if (index < 0)
    return _contours[_index];
  return _contours[index];
}
const cv::Mat &Particle::image(const int index) const {
  if (index < 0)
    return _images[_index];
  return _images[index];
}
const cv::Mat &Particle::rawImage(const int index) const {
  if (index < 0)
    return _raw_images[_index];
  return _raw_images[index];
}

const cv::Rect Particle::boundingRect() const {
  cv::Rect bounds = cv::boundingRect(_contours[0]);
  for (auto it = _contours.begin() + 1; it < _contours.end(); ++it) {
    cv::Rect rect = cv::boundingRect(*it);
    bounds = bounds | rect;
  }
  return bounds;
}

void Particle::update(const int frame_number,
                      const std::vector<cv::Point> &contour,
                      const cv::Mat &image, const cv::Mat &raw_image) {
  cv::Rect rect = cv::boundingRect(contour);
  rect &= cv::Rect(0, 0, image.cols, image.rows);

  _contours.push_back(contour);
  _frames.push_back(frame_number);
  _images.push_back(image(rect).clone());
  if (!raw_image.empty()) {
    _raw_images.push_back(raw_image(rect).clone());
  }

  double metric =
      calculate_selection_metric(contour, _images.back(), _metric_method);
  if (metric > _metric) {
    _metric = metric;
    _index = _frames.size() - 1;
  }
}

double calculate_selection_metric(const std::vector<cv::Point> &contour,
                                  cv::InputArray &image,
                                  ParticleFrameMetric method) {

  switch (method) {
  case METRIC_CENTER_WEIGHTED_INTENSITY: {
    cv::Mat mask, buffer;
    mask_for_contour(contour, mask);
    return image_center_weighted_intensity(image, mask, buffer);
  }
  case METRIC_AVERAGE_INTENSITY: {
    cv::Mat mask;
    mask_for_contour(contour, mask);
    return image_intensity(image, mask) / cv::contourArea(contour);
  }
  case METRIC_SHARPNESS: {
    cv::Mat buffer;
    return image_sharpness(image, buffer);
  }
  default:
    throw "unknown selection metric";
  }
}

long Particle::id_counter = 0;
