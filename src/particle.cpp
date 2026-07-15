#include "particle.hpp"

#include <numbers>
#include <numeric>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

Particle::Particle(const std::vector<cv::Point> &contour, const cv::Mat &image,
                   int frame_number, const cv::Mat &raw_image)
    : _id(id_counter++) {
  // moments for center and area
  _contours.push_back(contour);
  _frames.push_back(frame_number);
  _frame_index = 0;

  _moments = cv::moments(contour);

  _rect = cv::boundingRect(contour);

  // _rect -= cv::Point(_rect.size()) * 0.5;
  // _rect += _rect.size();
  _rect &= cv::Rect(0, 0, image.cols, image.rows);

  _image = image(_rect).clone();
  // _mask = cv::Mat::zeros(_image.rows, _image.cols, CV_8U);
  // cv::drawContours(_mask, {contour}, 0, 255, -1, cv::LINE_8, cv::noArray(),
  // 0,
  //                  -_rect.tl());
};

const std::vector<cv::Point> &Particle::contour() const {
  return _contours[_frame_index];
}

int Particle::frameCount() const { return _frames.size(); }

const int Particle::frameNumber() const { return _frames[_frame_index]; }

const long Particle::id() const { return _id; }

const cv::Mat &Particle::image() const { return _image; }

const cv::Mat &Particle::rawImage() const { return _image_raw; }

// Calculated
double Particle::area() const { return _moments.m00; };

double Particle::aspect() const {
  double aspect = _min_area_rect.size.aspectRatio();
  if (aspect > 1.0) {
    aspect = 1.0 / aspect;
  }
  return aspect;
}

cv::Point2f Particle::center() const {
  return cv::Point2f(_moments.m10 / _moments.m00, _moments.m01 / _moments.m00);
}

double Particle::circularity() const {
  auto perim = cv::arcLength(_contour, true);
  return 4.0 * std::numbers::pi * area() / std::pow(perim, 2);
}

double Particle::convexity() const {
  std::vector<cv::Point> hull;
  cv::convexHull(_contour, hull);
  return area() / cv::contourArea(hull);
};

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

const double Particle::circularEquvalentDiameter() const {
  // moments.m00 = area()
  return std::sqrt((4.0 * _moments.m00) / std::numbers::pi);
}

double Particle::intensity() const {
  cv::Mat intensity = cv::Mat::zeros(_image.rows, _image.cols, CV_8U);
  _image.copyTo(intensity, _mask);
  return cv::sum(intensity)[0];
};

const double Particle::maximumWidth() const {
  // longest length of min area rect
  return std::max(_min_area_rect.size.width, _min_area_rect.size.height);
}

const double Particle::minimumWidth() const {
  // shortest length of min area rect
  return std::min(_min_area_rect.size.width, _min_area_rect.size.height);
}

const double Particle::perimeter() const {
  return cv::arcLength(contour(), true);
}

double Particle::radius() const {
  const cv::Point2f pc = center();
  const std::vector<cv::Point> &c = contour();
  double dist = std::accumulate(c.begin(), c.end(), 0.0,
                                [&pc](double sum, const cv::Point2f &p) {
                                  return sum + cv::norm(p - pc);
                                });
  return dist / c.size();
}

double Particle::sharpness() const {
  cv::Mat laplace;
  cv::Laplacian(_image, laplace, CV_32F);
  cv::Scalar mu, sigma;
  cv::meanStdDev(laplace, mu, sigma);
  return sigma[0];
}

void Particle::addFrames(const int count) { _frame_count += count; }

void Particle::setRawImage(const cv::Mat &frame) {
  _image_raw = frame(_rect).clone();
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

long Particle::id_counter = 0;
