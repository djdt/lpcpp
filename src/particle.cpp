#include "particle.hpp"
#include "tracy/Tracy.hpp"

#include <opencv2/imgcodecs.hpp>

#include <execution>

Particle::Particle(const std::vector<cv::Point> &contour, const cv::Mat &frame,
                   int frame_number, int id)
    : _contour(contour), _frame(frame_number), _frame_count(1), _id(id) {
  ZoneScoped;

  // moments for center and area
  _moments = cv::moments(_contour);

  _rect = cv::boundingRect(_contour);

  _rect -= cv::Point(_rect.size()) * 0.5;
  _rect += _rect.size();
  _rect &= cv::Rect(0, 0, frame.cols, frame.rows);

  _image = frame(_rect).clone();
  _mask = cv::Mat::zeros(_image.rows, _image.cols, CV_8U);
  cv::drawContours(_mask, {_contour}, 0, 255, -1, cv::LINE_8, cv::noArray(), 0,
                   -_rect.tl());
};

const std::vector<cv::Point> &Particle::contour() const { return _contour; }

int Particle::frame_count() const { return _frame_count; }

int Particle::frame_number() const { return _frame; }

const cv::Mat &Particle::image() const { return _image; }

int Particle::id() const { return _id; }

// Calculated
double Particle::area() const { return _moments.m00; };

double Particle::aspect() const {
  ZoneScoped;
  cv::RotatedRect rect = cv::minAreaRect(_contour);
  double aspect = rect.size.aspectRatio();
  if (aspect > 1.0) {
    aspect = 1.0 / aspect;
  }
  return aspect;
}

cv::Point2f Particle::center() const {
  ZoneScoped;
  return cv::Point2f(_moments.m10 / _moments.m00, _moments.m01 / _moments.m00);
}

double Particle::circularity() const {
  ZoneScoped;
  auto perim = cv::arcLength(_contour, true);
  return 4.0 * std::numbers::pi * area() / std::pow(perim, 2);
}

double Particle::convexity() const {
  ZoneScoped;
  std::vector<cv::Point> hull;
  cv::convexHull(_contour, hull);
  return area() / cv::contourArea(hull);
};

const std::vector<cv::Point> Particle::imageContour() const {
  ZoneScoped;
  std::vector<cv::Point> contour;
  contour.reserve(_contour.size());
  std::transform(_contour.begin(), _contour.end(), std::back_inserter(contour),
                 [&](const cv::Point &p) { return p - _rect.tl(); });
  return contour;
}

const double Particle::centerWeightedIntensity() const {
  ZoneScoped;
  cv::Mat weights(_image.rows, _image.cols, CV_32F);
  cv::distanceTransform(_mask, weights, cv::DIST_L2, cv::DIST_MASK_3);
  cv::multiply(weights, _image, weights, 1.0 / cv::sum(weights)[0]);

  return cv::sum(weights)[0];
}

double Particle::intensity() const {
  ZoneScoped;
  cv::Mat intensity = cv::Mat::zeros(_image.rows, _image.cols, CV_8U);
  _image.copyTo(intensity, _mask);
  return cv::sum(intensity)[0];
};

double Particle::radius() const {
  ZoneScoped;
  const cv::Point2f c = center();
  double dist = std::accumulate(
      _contour.begin(), _contour.end(), 0.0,
      [&c](double sum, const cv::Point2f &p) { return sum + cv::norm(p - c); });
  return dist / _contour.size();
}

// double Particle::radiusAtQuantile(const double quantile) const {
//   cv::Mat values;
//   std::vector<cv::Point> points;
//   _image.copyTo(values, _mask);
//   cv::findNonZero(values, points);
//
//   auto loc = points.begin() + points.size() * quantile;
//   std::nth_element(points.begin(), loc, points.end(),
//                    [&](const cv::Point &a, const cv::Point &b) {
//                      return values.at<float>(a) < values.at<float>(b);
//                    });
//   auto q = values.at<float>(points[points.size() * quantile]);
//
//   cv::Mat qmask;
//   cv::threshold(values, qmask, q, 255, cv::THRESH_BINARY);
//   qmask.convertTo(qmask, CV_8U);
//
//   std::vector<std::vector<cv::Point>> contours;
//   cv::findContours(qmask, contours, cv::RETR_EXTERNAL,
//   cv::CHAIN_APPROX_SIMPLE,
//                    _rect.tl());
//
//   if (contours.size() == 0)
//     return 0;
//
//   const cv::Point2f c = center();
//   double dist = std::accumulate(
//       contours[0].begin(), contours[0].end(), 0.0,
//       [&c](double sum, const cv::Point2f &p) { return sum + cv::norm(p - c);
//       });
//   return dist / contours[0].size();

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
  ZoneScoped;
  return cv::norm(center() - b.center()) - radius() - b.radius() <
         edge_distance;
};

void filter_particles(std::vector<Particle> &particles,
                      struct filter_args args) {
  ZoneScoped;
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
