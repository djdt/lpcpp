#include "particle.hpp"
#include "contours.hpp"
#include "cpuproc.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

Particle::Particle(const int frame_number,
                   const std::vector<cv::Point> &contour, const cv::Mat &image,
                   const cv::Mat &raw_image, const ParticleMetric method)
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
const int Particle::lastFrame() const { return _frames.back(); }
const long Particle::id() const { return _id; }

const int Particle::frame() const { return _frames[_index]; }
const std::vector<cv::Point> &Particle::contour() const {
  return _contours[_index];
}
const cv::Mat &Particle::image() const { return _images[_index]; }
const cv::Mat &Particle::rawImage() const { return _raw_images[_index]; }

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
                                  ParticleMetric method) {
  switch (method) {
  case CENTER_WEIGHTED_INTENSITY: {
    cv::Mat mask, weights;
    mask_for_contour(contour, mask);
    return image_center_weighted_intensity(image, mask, weights);
  }
  case INTENSITY: {
    cv::Mat mask;
    mask_for_contour(contour, mask);
    return image_intensity(image, mask);
  }
  case SHARPNESS: {
    cv::Mat laplace;
    return image_sharpness(image, laplace);
  }
  default:
    throw "unknown selection metric";
  }
}

long Particle::id_counter = 0;
