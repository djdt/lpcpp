#include <algorithm>
#include <execution>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "particle.hpp"
#include "util.hpp"

cv::Vec3f find_capillary(cv::InputArray &input) {
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(input, circles, cv::HOUGH_GRADIENT, 1.0,
                   static_cast<float>(input.rows()) / 2.f, 50, 5,
                   input.rows() / 4, input.rows());

  if (circles.size() == 0) {
    return cv::Vec3f(0.f, 0.f, 0.f);
  }
  return circles[0];
}

void unsharp_mask(cv::InputArray &image, cv::OutputArray &output,
                  double alpha = 1.0) {
  cv::UMat sobelx, sobely, mag;
  cv::Sobel(image, sobelx, CV_32F, 1, 0, 3);
  cv::Sobel(image, sobely, CV_32F, 0, 1, 3);
  cv::magnitude(sobelx, sobely, mag);
  cv::addWeighted(image, 1.0 + alpha, mag, -alpha, 0, output);
}

void update_background(cv::InputArray &frame, cv::InputOutputArray &mean,
                       cv::InputOutputArray &var, int pos) {
  double weight = 1.0 / static_cast<double>(pos);

  cv::addWeighted(frame, weight, mean, 1.0 - weight, 0.0, mean, CV_32F);

  cv::UMat tmp;
  cv::subtract(frame, mean, tmp, cv::noArray(), CV_32F);
  cv::pow(tmp, 2.0, tmp);
  cv::addWeighted(tmp, weight, var, 1.0 - weight, 0.0, var);
}

bool init_background(cv::VideoCapture &cap, cv::InputOutputArray &mean,
                     cv::InputOutputArray &var, int frame_count) {
  int frame_pos = 0;
  cap.set(cv::CAP_PROP_POS_FRAMES, frame_pos);

  cv::UMat frame;

  auto start_time = std::chrono::system_clock::now();

  while (frame_pos++ < frame_count) {
    cap.read(frame);
    if (frame.empty()) {
      std::cerr << "video does not contain enough background frames"
                << std::endl;
      return true;
    }

    // update the background accumulated mean and variance
    update_background(frame, mean, var, frame_pos);

    // update progress
    if (frame_pos % 100 == 0) {
      double fps;
      auto remaining =
          get_remaining_time(start_time, frame_pos, frame_pos, fps);

      std::cout << "\t...processing background :: frame " << frame_pos << "/"
                << frame_count << " @ ";
      std::cout << std::setw(3) << static_cast<int>(fps) << " FPS, ";
      std::cout << std::format("{:%T}", remaining) << " remaining.\r"
                << std::flush;
    }
  }
  std::cout << std::endl;
  return false;
}

void find_particles(cv::InputArray &frame, cv::InputArray &mean,
                    cv::InputArray &var, const double zscore,
                    cv::InputArray &mask, const double unsharp_alpha,
                    std::vector<Particle> &particles, const int current_frame) {
  // calculate the difference between frame and mean
  cv::UMat diff;
  frame.copyTo(diff);
  diff.convertTo(diff, CV_32F);
  // frame.getMat().convertTo(diff, CV_32F);
  cv::subtract(diff, mean, diff);
  cv::multiply(diff, -1.f, diff);

  // median blur
  cv::medianBlur(diff, diff, 5);

  // sharpen
  if (unsharp_alpha > 0.0)
    unsharp_mask(diff, diff, unsharp_alpha);

  // mask differences below x std deviations
  cv::UMat std;
  cv::sqrt(var, std);
  cv::multiply(std, zscore, std);

  cv::UMat thresh = cv::UMat(frame.rows(), frame.cols(), CV_8U);
  cv::compare(diff, std, thresh, cv::CMP_GT);

  cv::bitwise_and(thresh, mask, thresh);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(thresh, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  const cv::Mat cpu_diff = diff.getMat(cv::ACCESS_READ);
  particles.reserve(contours.size());
  std::transform(contours.begin(), contours.end(),
                 std::back_inserter(particles),
                 [&](const std::vector<cv::Point> &contour) {
                   return Particle(contour, cpu_diff, current_frame);
                 });
}

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
                  it_new->addFrames(old.frameCount()); // inherit old particle count
                  return true; // old is removed
                } else {
                  // remove new
                  size_t idx = std::distance(new_particles.begin(), it_new);
                  if (remove_new_at.size() == 0 or
                      remove_new_at.back() != idx) {
                    remove_new_at.push_back(idx);
                  }
                  old.addFrames(1);
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
