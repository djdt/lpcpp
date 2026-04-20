#include "particle.hpp"

#include <execution>
#include <numbers>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

Particle::Particle(const std::vector<cv::Point> &contour, const cv::Mat &frame,
                   int frame_number, int id)
    : _contour(contour), _frame(frame_number), _frame_count(1), _id(id) {
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

  _min_area_rect = cv::minAreaRect(_contour);
};

const std::vector<cv::Point> &Particle::contour() const { return _contour; }

int Particle::frameCount() const { return _frame_count; }

const int Particle::frameNumber() const { return _frame; }

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
  return cv::arcLength(_contour, true);
}

double Particle::radius() const {
  const cv::Point2f c = center();
  double dist = std::accumulate(
      _contour.begin(), _contour.end(), 0.0,
      [&c](double sum, const cv::Point2f &p) { return sum + cv::norm(p - c); });
  return dist / _contour.size();
}

double Particle::sharpness() const {
  cv::Mat laplace;
  cv::Laplacian(_image, laplace, CV_32F);
  cv::Scalar mu, sigma;
  cv::meanStdDev(laplace, mu, sigma);
  return sigma[0];
}

void Particle::addFrame() { _frame_count++; }

void Particle::addRawImage(const cv::Mat &frame) {
  _image_raw = frame(_rect).clone();
}

// Comparison
bool Particle::isClose(const Particle &b, double edge_distance) {
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
            if (args.min_sharpness != args.max_sharpness) {
              double sharpness = p.sharpness();
              if (sharpness < args.min_sharpness or
                  sharpness > args.max_sharpness) {
                return true;
              }
            }
            return false;
          }),
      particles.end());
}

void filter_existing_particles(
    std::vector<Particle> &old_particles, std::vector<Particle> &new_particles,
    const std::function<bool(const Particle &, const Particle &)> comparison,
    const double edge_distance) {
  std::vector<size_t> remove_new_at;
  old_particles.erase(
      std::remove_if(
          std::execution::seq, old_particles.begin(), old_particles.end(),
          [&](Particle &old) {
            for (auto it_new = new_particles.begin();
                 it_new != new_particles.end(); ++it_new) {

              if (it_new->isClose(old, edge_distance)) {

                if (comparison(*it_new, old)) {
                  it_new->addFrame();
                  return true; // old is removed
                } else {
                  // remove new
                  size_t idx = std::distance(new_particles.begin(), it_new);
                  if (remove_new_at.size() == 0 or
                      remove_new_at.back() != idx) {
                    remove_new_at.push_back(idx);
                  }
                  old.addFrame();
                  return false;
                }
              }
            }
            return false;
          }),
      old_particles.end());

  // sort and remove non-unqiue indicies
  std::sort(remove_new_at.begin(), remove_new_at.end());
  auto last = std::unique(remove_new_at.begin(), remove_new_at.end());
  remove_new_at.erase(last, remove_new_at.end());

  new_particles.erase(remove_indices(new_particles.begin(), new_particles.end(),
                                     remove_new_at.begin(),
                                     remove_new_at.end()),
                      new_particles.end());
}
