#pragma once

#include <opencv2/core.hpp>
#include <vector>

class Particle {
private:
  std::vector<cv::Point> _contour;
  cv::Mat _image;
  cv::Mat _mask;
  cv::Rect _rect;

  int _id;
  int _frame;
  int _frame_count;

  cv::Moments _moments;

public:
  Particle(const std::vector<cv::Point> &contour, const cv::Mat &frame,
           int frame_number, int id);

  double area() const;
  double aspect() const;
  cv::Point2f center() const;
  const std::vector<cv::Point> &contour() const;
  const double centerWeightedIntensity() const;
  double circularity() const;
  double convexity() const;
  int frame_number() const;
  int frame_count() const;
  const cv::Mat &image() const;
  const std::vector<cv::Point> imageContour() const;
  double intensity() const;
  int id() const;
  double radius() const;
  double sharpness() const;

  void addFrame();
  bool is_close(const Particle &b, double edge_distance = 0.0);
};

struct filter_args {
  double min_area = 0.0;
  double max_area = 0.0;
  double min_aspect = 0.0;
  double max_aspect = 0.0;
  double min_circularity = 0.0;
  double max_circularity = 0.0;
  double min_convexity = 0.0;
  double max_convexity = 0.0;
  double min_radius = 0.0;
  double max_radius = 0.0;
  double min_intensity = 0.0;
  double max_intensity = 0.0;
};

template <typename Iter, typename IdxIter>
auto remove_indices(Iter begin, Iter end, IdxIter indices_begin,
                    IdxIter indices_end) -> Iter {

  while (indices_end != indices_begin) {
    --indices_end;
    auto pos = begin + *indices_end;
    std::move(std::next(pos), end--, pos);
  }
  return end;
}

/* Filters particles by proterty, removing failing from the vector.
 * To enabled a filter pass different min and max values. */
void filter_particles(std::vector<Particle> &particles, struct filter_args);

void filter_existing_particles(
    std::vector<Particle> &old_particles, std::vector<Particle> &new_particles,
    const std::function<bool(const Particle &, const Particle &)> comparision,
    const double edge_distance = 20.0);
