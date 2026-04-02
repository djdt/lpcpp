#include "particle.hpp"

#include <opencv2/imgcodecs.hpp>

#include <execution>

Particle::Particle(const std::vector<cv::Point> &contour, const cv::Mat &frame,
                   int frame_number, int id, int image_scale)
    : _contour(contour), _frame(frame_number), _frame_count(1), _id(id) {
  // moments for center and area
  _moments = cv::moments(_contour);
  // cv::Point2f center =
  //     cv::Point2f(_moments.m10 / _moments.m00, _moments.m01 / _moments.m00);

  _rect = cv::boundingRect(_contour);
  _rect -= cv::Point(_rect.size()) * image_scale / 2;
  _rect += _rect.size() * image_scale;
  _rect &= cv::Rect(0, 0, frame.cols, frame.rows);

  _image = frame(_rect).clone();

  // search again at a percentile
  double max;
  cv::minMaxIdx(_image, nullptr, &max);

  // mask off invalid pixels
  // std::vector<std::vector<cv::Point>> c = {_contour};
  // cv::Mat mask = cv::Mat::ones(2, _image.size, CV_8U);
  // cv::drawContours(mask, c, 0, 0, -1, cv::LINE_8, cv::noArray(), 1,
  // -_rect.tl()); _image.setTo(255, mask);
};

const std::vector<cv::Point> &Particle::contour() const { return _contour; }

int Particle::frame_count() const { return _frame_count; }

int Particle::frame_number() const { return _frame; }

const cv::Mat &Particle::image() const { return _image; }

int Particle::id() const { return _id; }

// Calculated
double Particle::area() const { return _moments.m00; };

double Particle::aspect() const {
  cv::RotatedRect rect = cv::minAreaRect(_contour);
  double aspect = rect.size.aspectRatio();
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

const cv::Mat Particle::imageMask() const {
  cv::Mat mask = cv::Mat::zeros(2, _image.size, CV_8U);
  std::vector<std::vector<cv::Point>> contours = {_contour};
  cv::drawContours(mask, contours, 0, 255, -1, cv::LINE_8, cv::noArray(), 0,
                   -_rect.tl());
  return mask;
}

const double Particle::centerWeightedIntensity() const {
  cv::Mat mask = imageMask();
  cv::Mat weights(_image.size.dims(), _image.size, CV_32F);
  cv::distanceTransform(mask, weights, cv::DIST_L2, cv::DIST_MASK_3);
  cv::multiply(weights, _image, weights, 1.0 / cv::sum(weights)[0]);

  return cv::sum(weights)[0];
}

double Particle::intensity() const {
  cv::Mat intensity = cv::Mat::zeros(2, _image.size, CV_8U);
  _image.copyTo(intensity, imageMask());
  return cv::sum(intensity)[0];
};

double Particle::radius() const {
  const cv::Point2f c = center();
  double dist = std::accumulate(
      _contour.begin(), _contour.end(), 0.0,
      [&c](double sum, const cv::Point2f &p) { return sum + cv::norm(p - c); });
  return dist / _contour.size();
}

double Particle::radiusAtQuantile(const double quantile) const {
  cv::Mat values;
  std::vector<cv::Point> points;
  _image.copyTo(values, imageMask());
  cv::findNonZero(values, points);

  auto loc = points.begin() + points.size() * quantile;
  std::nth_element(points.begin(), loc, points.end(),
                   [&](const cv::Point &a, const cv::Point &b) {
                     return values.at<float>(a) < values.at<float>(b);
                   });
  auto q = values.at<float>(points[points.size() * quantile]);

  cv::Mat qmask;
  cv::threshold(values, qmask, q, 255, cv::THRESH_BINARY);
  qmask.convertTo(qmask, CV_8U);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(qmask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE,
                   _rect.tl());

  if (contours.size() == 0)
    return 0;

  const cv::Point2f c = center();
  double dist = std::accumulate(
      contours[0].begin(), contours[0].end(), 0.0,
      [&c](double sum, const cv::Point2f &p) { return sum + cv::norm(p - c); });
  return dist / contours[0].size();
}

double Particle::sharpness() const {
  cv::Mat laplace;
  cv::Laplacian(_image, laplace, CV_32F);
  cv::Scalar mu, sigma;
  cv::meanStdDev(laplace, mu, sigma);
  return sigma[0];
}

void Particle::addFrame() { _frame_count++; }

// Comparison
bool Particle::is_close(const Particle &b, double edge_distance) {
  return cv::norm(center() - b.center()) - radius() - b.radius() <
         edge_distance;
};

void filter_particles(std::vector<Particle> &particles,
                      struct filter_args args) {
  particles.erase(
      std::remove_if(
          std::execution::par, particles.begin(), particles.end(),
          [=](Particle &p) {
            if (args.min_area != args.max_area) {
              double area = p.area();
              if (area < args.min_area or area > args.max_area) {
                return true;
              }
            }
            if (args.min_aspect != args.max_aspect) {
              double aspect = p.aspect();
              if (aspect < args.min_aspect or aspect > args.max_aspect) {
                return true;
              }
            }
            if (args.min_circularity != args.max_circularity) {
              double circularity = p.circularity();
              if (circularity < args.min_circularity or
                  circularity > args.max_circularity) {
                return true;
              }
            }
            if (args.min_convexity != args.max_convexity) {
              double convexity = p.convexity();
              if (convexity < args.min_convexity or
                  convexity > args.max_convexity) {
                return true;
              }
            }
            if (args.min_radius != args.max_radius) {
              double radius = p.radius();
              if (radius < args.min_radius or radius > args.max_radius) {
                return true;
              }
            }
            if (args.min_intensity != args.max_intensity) {
              double intensity = p.intensity();
              if (intensity < args.min_intensity or
                  intensity > args.max_intensity) {
                return true;
              }
            }
            return false;
          }),
      particles.end());
}
