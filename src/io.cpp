#include "io.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <iterator>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "contours.hpp"
#include "cpuproc.hpp"
#include "particle.hpp"

void draw_particles_on_frame(cv::InputArray &input,
                             cv::InputOutputArray &output,
                             std::vector<Particle> &particles) {

  output.createSameSize(input, CV_8UC3);
  cv::cvtColor(input, output, cv::COLOR_GRAY2BGR);

  auto color = cv::Scalar(0, 0, 255);
  std::vector<std::vector<cv::Point>> contours;
  contours.reserve(particles.size());

  std::transform(particles.begin(), particles.end(),
                 std::back_inserter(contours),
                 [](const Particle &p) { return p.contour(); });

  cv::drawContours(output, contours, -1, color, 1.0, 8);
}

void write_particle_header(std::ofstream &ofs) {
  ofs << "id,frame,frame_count,area,aspect,circular_equivalent_diameter,"
         "circularity,convexity,intensity,maximum_width,minimum_width,"
         "perimeter,sharpness,x,y"
      << std::endl;
}

void write_particle_data(const std::vector<Particle> &particles,
                         std::ofstream &ofs) {
  cv::Mat mask;
  for (auto it = particles.begin(); it != particles.end(); ++it) {
    const std::vector<cv::Point> &contour = it->contour();
    cv::Moments moments = cv::moments(contour);
    mask_for_contour(contour, mask);

    ofs << it->id() << ",";
    ofs << it->frame() << ",";
    ofs << it->frameCount() << ",";
    ofs << moments.m00 << ",";
    ofs << contour_aspect(contour) << ",";
    ofs << contour_circular_equivalent_diameter(contour, moments.m00) << ",";
    ofs << contour_circularity(contour, moments.m00) << ",";
    ofs << contour_convexity(contour, moments.m00) << ",";
    ofs << image_intensity(it->image(), mask) << ",";
    ofs << contour_maximum_feret(contour) << ",";
    ofs << contour_minimum_feret(contour) << ",";
    ofs << cv::arcLength(contour, true) << ",";
    ofs << image_sharpness(it->image(), mask) << ",";
    ofs << moments.m10 / moments.m00 << ",";
    ofs << moments.m01 / moments.m00 << std::endl;
  }
}

bool save_particle_image(const Particle &particle,
                         const std::filesystem::path &path) {
  auto color = cv::Scalar(0, 0, 255);
  cv::Mat image, raw_image, mask;
  particle.image().convertTo(image, CV_8U);
  particle.rawImage().convertTo(raw_image, CV_8U);
  mask_for_contour(particle.contour(), mask);
  const cv::Mat src[] = {raw_image, image, mask};
  cv::Mat dst;
  cv::merge(src, 3, dst);
  if (not cv::imwrite(path.string(), dst)) {
    std::cerr << "failed to save " << path << std::endl;
    return true;
  }
  return false;
}

bool save_particle_point_data_vtk(const Particle &particle,
                                  const std::filesystem::path &path) {
  // find extents
  cv::Rect extents;
  std::vector<cv::Rect> rects;
  for (const auto &c : particle.contour()) {
    cv::Rect rect = cv::boundingRect(c);
    extents = extents | rect;
    rects.push_back(rect);
  }
  std::ofstream ofs(path);
  ofs << "<VTKFile type=\"ImageData\" version=\"0.1\" "
         "byte_order=\"LittleEndian\">\n";
  ofs << "\t<ImageData WholeExtent=\"" << extents.x << " "
      << extents.x + extents.width << " " << extents.y << " "
      << extents.y + extents.height << " " << 0 << " " << particle.frameCount()
      << "\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n";

  for (size_t i = 0; i < particle.frameCount(); ++i) {
    cv::Mat layer = cv::Mat::zeros(extents.size(), CV_8U);
    layer(rects[i] - extents.tl()) = images[i];
    ofs << "\t\t<Piece Extent=\"" << rects[i].x << " "
        << rects[i].x + rects[i].width << " " << rects[i].y << " "
        << rects[i].y + rects[i].height << " " << i << " " << i + 1 << "\">\n";
    ofs << "\t\t\t<PointData>"
  }
}
