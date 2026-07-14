#include <filesystem>
#include <fstream>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "particle.hpp"

void write_particle_header(std::ofstream &ofs) {
  ofs << "id,frame,frame_count,area,aspect,circular_equivalent_diameter,"
         "circularity,convexity,intensity,maximum_width,minimum_width,"
         "perimeter,radius,sharpness,x,y"
      << std::endl;
}

void write_particle_data(const std::vector<Particle> &particles,
                         std::ofstream &ofs) {
  for (auto it = particles.begin(); it != particles.end(); ++it) {
    ofs << it->id() << ",";
    ofs << it->frameNumber() << ",";
    ofs << it->frameCount() << ",";
    ofs << it->area() << ",";
    ofs << it->aspect() << ",";
    ofs << it->circularEquvalentDiameter() << ",";
    ofs << it->circularity() << ",";
    ofs << it->convexity() << ",";
    ofs << it->intensity() << ",";
    ofs << it->maximumWidth() << ",";
    ofs << it->minimumWidth() << ",";
    ofs << it->perimeter() << ",";
    ofs << it->radius() << ",";
    ofs << it->sharpness() << ",";
    ofs << it->center().y << ",";
    ofs << it->center().x << std::endl;
  }
}

bool write_particle_images(const std::vector<Particle> &particles,
                           const std::filesystem::path &output_dir) {
  auto color = cv::Scalar(0, 0, 255);
  for (auto it = particles.begin(); it != particles.end(); ++it) {
    auto out = output_dir / std::to_string(it->id()).append(".png");
    cv::Mat image, raw_image;
    it->image().convertTo(image, CV_8U);
    it->rawImage().convertTo(raw_image, CV_8U);

    cv::Mat bgr = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    cv::insertChannel(image, bgr, 1);
    cv::insertChannel(raw_image, bgr, 0);

    cv::Mat fill = cv::Mat::zeros(image.rows, image.cols, CV_8U);
    // cv::polylines(rgb, it->imageContour(), -1, color, 1.0, 8);
    cv::fillPoly(fill, it->imageContour(), 255);
    cv::insertChannel(fill, bgr, 2);
    if (not cv::imwrite(out.string(), bgr)) {
      std::cerr << "failed to save " << out << std::endl;
      return true;
    }
  }
  return false;
}
