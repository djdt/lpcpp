#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "particle.hpp"

void write_particle_header(std::ofstream &ofs) {
  ofs << "id,frame,frame_count,area,aspect,circularity,convexity,intensity,"
         "radius,x,y"
      << std::endl;
}
void write_particle_data(const std::vector<Particle> &particles,
                         std::ofstream &ofs) {
  for (auto it = particles.begin(); it != particles.end(); ++it) {
    ofs << it->id() << "," << it->frame_number() << "," << it->frame_count()
        << ",";
    ofs << it->area() << "," << it->aspect() << "," << it->circularity() << ",";
    ofs << it->convexity() << "," << it->intensity() << "," << it->radius()
        << ",";
    ofs << it->center().x << "," << it->center().y << std::endl;
  }
}
bool export_particle_images(const std::vector<Particle> &particles,
                            const std::filesystem::path &output_dir) {
  auto color = cv::Scalar(0, 0, 255);
  for (auto it = particles.begin(); it != particles.end(); ++it) {
    auto out = output_dir / std::to_string(it->id()).append(".png");
    cv::Mat image;
    it->image().convertTo(image, CV_8U);
    cv::Mat rgb = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    cv::insertChannel(image, rgb, 1);

    cv::Mat fill = cv::Mat::zeros(image.rows, image.cols, CV_8U);
    // cv::polylines(rgb, it->imageContour(), -1, color, 1.0, 8);
    cv::fillPoly(fill, it->imageContour(), 255);
    cv::insertChannel(fill, rgb, 2);
    if (not cv::imwrite(out, rgb)) {
      std::cerr << "failed to save " << out << std::endl;
      return true;
    }
  }
  return false;
}

template <typename Iter>
void draw_current_frame(cv::InputArray &_frame, const Iter &begin,
                        const Iter &end) {
  cv::Mat frame;
  if (_frame.isGpuMat()) {
    cv::cuda::GpuMat _gpu_frame = _frame.getGpuMat();
    _gpu_frame.download(frame);
  } else {
    frame = _frame.getMat();
  }
  cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
  auto color = cv::Scalar(0, 0, 255);
  int decay = 255 / particle_frames;
  for (auto it = particles.rbegin(); it != particles.rend(); ++it) {
    contours.resize(it->size());
    std::transform(std::execution::par, it->begin(), it->end(),
                   contours.begin(),
                   [](const Particle &p) { return p.contour(); });
    cv::drawContours(frame, contours, -1, color, 1.0, 8);
    color[2] -= decay;
  }
  // get the filtered contours
  cv::imshow("frame", frame);
  diff.convertTo(diff, -1, 1.0 / 255.0, 0.5);
  cv::imshow("diff", diff);

  int key = cv::waitKey(5000);
  if (key == 'q') {
    break;
  }
}
