#include <filesystem>
#include <fstream>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "particle.hpp"

void write_particle_header(std::ofstream &ofs) {
  ofs << "id,frame,frame_count,area,aspect,circularity,convexity,intensity,"
         "radius,sharpness,x,y"
      << std::endl;
}
void write_particle_data(const std::vector<Particle> &particles,
                         std::ofstream &ofs) {
  for (auto it = particles.begin(); it != particles.end(); ++it) {
    ofs << it->id() << "," << it->frame_number() << "," << it->frame_count()
        << "," << it->area() << "," << it->aspect() << "," << it->circularity()
        << "," << it->convexity() << "," << it->intensity() << ","
        << it->radius() << "," << it->sharpness() << "," << it->center().x
        << "," << it->center().y << std::endl;
  }
}
bool write_particle_images(const std::vector<Particle> &particles,
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
    if (not cv::imwrite(out.string(), rgb)) {
      std::cerr << "failed to save " << out << std::endl;
      return true;
    }
  }
  return false;
}

void read_filter_config(std::string path, filter_args &args) {
  std::fstream ifs(path);
  std::string line;
  while (!ifs.eof()) {
    ifs >> line;
    if (line == "area")
      ifs >> args.min_area >> args.max_area;
    else if (line == "aspect")
      ifs >> args.min_aspect >> args.max_aspect;
    else if (line == "circularity")
      ifs >> args.min_circularity >> args.max_circularity;
    else if (line == "convexity")
      ifs >> args.min_convexity >> args.max_convexity;
    else if (line == "radius")
      ifs >> args.min_radius >> args.max_radius;
    else if (line == "intensity")
      ifs >> args.min_intensity >> args.max_intensity;
    else if (line == "sharpness")
      ifs >> args.min_sharpness >> args.max_sharpness;
    else {
      std::cout << "unknown filter value '" << line << "'" << std::endl;
    }
  }
}
