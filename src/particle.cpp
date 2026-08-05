#include "particle.hpp"

#include "contours.hpp"
#include "cpuproc.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/geometry/2d.hpp>
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
  // _contour_moments.push_back(cv::moments(contour));

  _images.push_back(image(rect).clone());
  if (!raw_image.empty()) {
    _raw_images.push_back(raw_image(rect).clone());
  }

  // _kalman.init(4, 2);
  //
  // static float t_vals[4][4] = {{1.f, 0.f, 1.f, 0.f},
  //                              {0.f, 1.f, 0.f, 1.f},
  //                              {0.f, 0.f, 1.f, 0.f},
  //                              {0.f, 0.f, 0.f, 1.f}};
  //
  // _kalman.transitionMatrix = cv::Mat(4, 4, CV_32F, t_vals);
  // _kalman.measurementMatrix = cv::Mat::eye(2, 4, CV_32F);
  //
  // cv::setIdentity(_kalman.processNoiseCov, 1e-3);
  // cv::setIdentity(_kalman.measurementNoiseCov, 1e-2);
  // cv::setIdentity(_kalman.errorCovPost, 1.f);
  //
  // cv::Moments moments = cv::moments(contour);
  // _kalman.statePost = cv::Mat::zeros(4, 1, CV_32F);
  // _kalman.statePost.at<float>(0) = moments.m10 / moments.m00;
  // _kalman.statePost.at<float>(1) = moments.m01 / moments.m00;

  _metric = calculate_selection_metric(contour, _images.back(), _metric_method);
};

const int Particle::frameCount() const { return _frames.size(); }
const long Particle::id() const { return _id; }

const int Particle::lastFrame() const { return _frames.back(); }

const std::vector<cv::Point> &Particle::lastContour() const {
  return _contours.back();
}

const std::vector<cv::Point> &Particle::contour(const int index) const {
  if (index < 0)
    return _contours[_index];
  return _contours[index];
}

const int Particle::frame(const int index) const {
  if (index < 0)
    return _frames[_index];
  return _frames[index];
}

const cv::Mat &Particle::image(const int index) const {
  if (index < 0)
    return _images[_index];
  return _images[index];
}
// const cv::Moments &Particle::moments(const int index) const {
//   if (index < 0)
//     return _contour_moments[_index];
//   return _contour_moments[index];
// }
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
  // existing contour on this frame, merge using convex hull of both
  if (_frames.back() == frame_number) {
    _contours.back().insert(_contours.back().end(), contour.begin(),
                            contour.end());
    std::vector<cv::Point> hull;
    cv::convexHull(_contours.back(), hull);

    _contours.back() = hull;
    _frames.pop_back();
    _images.pop_back();
    if (!raw_image.empty())
      _raw_images.pop_back();
  } else {
    _contours.push_back(contour);
  }

  cv::Rect rect = cv::boundingRect(_contours.back());
  rect &= cv::Rect(0, 0, image.cols, image.rows);

  _frames.push_back(frame_number);
  _images.push_back(image(rect).clone());
  if (!raw_image.empty())
    _raw_images.push_back(raw_image(rect).clone());

  double metric = calculate_selection_metric(_contours.back(), _images.back(),
                                             _metric_method);
  if (metric > _metric) {
    _metric = metric;
    _index = _frames.size() - 1;
  }

  // cv::Point2f center = contour_center(_contours.back());
  // cv::Mat measurement = cv::Mat(2, 1, CV_32F);
  // measurement.at<float>(0) = center.x;
  // measurement.at<float>(1) = center.y;
  //
  // _kalman.predict();
  // _kalman.correct(measurement);
}

cv::Point2f Particle::position() const {
  return cv::Point2f(_kalman.statePost.at<float>(0),
                     _kalman.statePost.at<float>(1));
}

cv::Point2f Particle::velocity() const {
  return cv::Point2f(_kalman.statePost.at<float>(2),
                     _kalman.statePost.at<float>(3));
}

cv::Point2f Particle::predictedPosition(const int frame) const {
  cv::Mat prediction = _kalman.statePost.clone();
  for (int i = _frames.back(); i < frame; ++i) {
    cv::gemm(_kalman.transitionMatrix, prediction, 1.0, cv::noArray(), 0.0,
             prediction);
  }
  return cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));
}

std::vector<cv::Point> Particle::trajectory(const int frame_count) const {
  cv::Mat prediction = _kalman.statePost.clone();
  std::vector<cv::Point> positions;
  positions.reserve(frame_count + 1);
  positions.push_back(position());

  for (int i = 0; i < frame_count; ++i) {
    cv::gemm(_kalman.transitionMatrix, prediction, 1.0, cv::noArray(), 0.0,
             prediction);
    positions.push_back(
        cv::Point(prediction.at<float>(0), prediction.at<float>(1)));
  }
  return positions;
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
