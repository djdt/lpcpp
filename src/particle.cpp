#include "particle.hpp"
#include "contours.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

Particle::Particle(int frame_number, const std::vector<cv::Point> &contour,
                   const cv::Mat &image, const cv::Mat &raw_image)
    : _id(id_counter++), _index(0) {

  cv::Rect rect = cv::boundingRect(contour);
  rect &= cv::Rect(0, 0, image.cols, image.rows);

  _frames.push_back(frame_number);
  _contours.push_back(contour);
  _images.push_back(image(rect).clone());
  if (!raw_image.empty()) {
    _raw_images.push_back(raw_image(rect).clone());
  }

  _metric = metric();
};

const double Particle::metric() const {
  cv::Mat weights, mask;
  mask_for_contour(contour(), mask);
  cv::distanceTransform(mask, weights, cv::DIST_L2, cv::DIST_MASK_3, CV_32F);
  cv::multiply(weights, image(), weights, 1.0 / cv::sum(weights)[0]);
  return cv::sum(weights)[0];
}

const int Particle::frameCount() const { return _frames.size(); }
const long Particle::id() const { return _id; }

const int Particle::frame() const { return _frames[_index]; }
const std::vector<cv::Point> &Particle::contour() const {
  return _contours[_index];
}
const cv::Mat &Particle::image() const { return _images[_index]; }
const cv::Mat &Particle::rawImage() const { return _raw_images[_index]; }

const std::vector<cv::Point> Particle::imageContour() const {
  std::vector<cv::Point> contour;
  contour.reserve(_contour.size());
  std::transform(_contour.begin(), _contour.end(), std::back_inserter(contour),
                 [&](const cv::Point &p) { return p - _rect.tl(); });
  return contour;
}

const double Particle::centerWeightedIntensity() const {
  cv::Mat weights(_image.rows, _image.cols, CV_32F);
  cv::distanceTransform(_mask, weights, cv::DIST_L2, cv::DIST_MASK_3);
  cv::multiply(weights, _image, weights, 1.0 / cv::sum(weights)[0]);

  return cv::sum(weights)[0];
}

double Particle::intensity() const {
  cv::Mat intensity = cv::Mat::zeros(_image.rows, _image.cols, CV_8U);
  _image.copyTo(intensity, _mask);
  return cv::sum(intensity)[0];
};

double Particle::sharpness() const {
  cv::Mat laplace;
  cv::Laplacian(_image, laplace, CV_32F);
  cv::Scalar mu, sigma;
  cv::meanStdDev(laplace, mu, sigma);
  return sigma[0];
}

void Particle::update(const std::vector<cv::Point> &contour,
                      const cv::Mat &image, const int frame_number,
                      const cv::Mat &raw_image) {
  _contours.push_back(contour);
  _frames.push_back(frame_number);

  cv::Rect rect = cv::boundingRect(contour);
  cv::Mat mask = cv::Mat::zeros(rect.size(), CV_8U);

  double intensity = center_weighted_intensity(contour, image);
  if (intensity > _intensity) {
    _frame_index = _frames.size();
    cv::Rect rect = cv::boundingRect(contour);
    _image = image(rect).clone();
    if (~raw_image.empty())
      _image_raw = raw_image(rect).clone();
  }
}

// Comparison
bool Particle::isClose(const Particle &b, const double edge_distance) {
  return cv::norm(center() - b.center()) - radius() - b.radius() <
         edge_distance;
};

double calculate_selection_metric(const std::vector<cv::Point> &contour,
                                  cv::InputArray &image,
                                  particle_metric method) {
  switch (method) {
  case CENTER_WEIGHTED_INTENSITY:
    cv::Mat weights;
    cv::distanceTransform(_mask, weights, cv::DIST_L2, cv::DIST_MASK_3);
    cv::multiply(weights, _image, weights, 1.0 / cv::sum(weights)[0]);

    return cv::sum(weights)[0];
  case INTENSITY:
    break;
  case SHARPNESS:
    break;
  default:
    throw "unknown selection metric";
  }
}

long Particle::id_counter = 0;
