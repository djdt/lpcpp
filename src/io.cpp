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

class base64_encoder {
private:
  std::ostream &_ostr;
  char _buffer[3];
  size_t _count;

public:
  explicit base64_encoder(std::ostream &ostr) : _ostr(ostr), _count(0) {}

  ~base64_encoder() {
    if (_count > 0) {
      encode_chunk(_count);
    }
  }

  template <typename T>
  typename std::enable_if<std::is_arithmetic<T>::value, base64_encoder &>::type
  operator<<(const T &val) {
    const uchar *bytes = reinterpret_cast<const uchar *>(&val);
    for (size_t i = 0; i < sizeof(T); ++i) {
      _buffer[_count++] = bytes[i];
      if (_count == 3) {
        encode_chunk(3);
        _count = 0;
      }
    }
    return *this;
  }

private:
  void encode_chunk(size_t bytes) {
    static const std::string b64 =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    uchar b1 = _buffer[0];
    uchar b2 = (bytes > 1) ? _buffer[1] : 0;
    uchar b3 = (bytes > 2) ? _buffer[2] : 0;

    _ostr << b64[(b1 >> 2) & 0x3F];
    _ostr << b64[((b1 << 4) | (b2 >> 4)) & 0x3F];
    _ostr << (bytes > 1 ? b64[((b2 << 2) | (b3 >> 6)) & 0x3F] : '=');
    _ostr << (bytes > 2 ? b64[b3 & 0x3F] : '=');
  }
};

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
  cv::Rect bounds = particle.boundingRect();
  std::ofstream ofs(path);

  ofs << "<VTKFile type=\"RectilinearGrid\" version=\"0.1\" "
         "byte_order=\"LittleEndian\">\n";
  ofs << "\t<RectilinearGrid WholeExtent=\"0 " << bounds.width << " 0 "
      << bounds.height << " 0 " << particle.frameCount() << "\">\n";
  ofs << "\t\t<Piece Extent=\"0 " << bounds.width << " 0 " << bounds.height
      << " 0 " << particle.frameCount() << "\">\n";
  ofs << "\t\t\t<CellData Scalars=\"processed\">\n";
  ofs << "\t\t\t\t<DataArray type=\"Float32\" Name=\"processed\" "
         "format=\"binary\">\n";

  for (size_t z = 0; z < particle.frameCount(); ++z) {
    auto b64enc = base64_encoder(ofs);

    const cv::Mat &image = particle.image(z);
    cv::Rect rect = cv::boundingRect(particle.contour(z));
    cv::Point offset = rect.tl() - bounds.tl();

    for (size_t y = 0; y < bounds.height; ++y) {
      for (size_t x = 0; x < bounds.width; ++x) {
        size_t sx = x - offset.x;
        size_t sy = y - offset.y;
        if (sx >= 0 && sx < image.cols && sy >= 0 && sy < image.rows) {
          b64enc << image.at<float>(sy, sx);
        } else {
          b64enc << 0.f;
        }
      }
    }
  }
  ofs << "\n";

  ofs << "\t\t\t\t</DataArray>\n";
  ofs << "\t\t\t</PointData>\n";

  ofs << "\t\t\t<Coordinates>\n";
  ofs << "\t\t\t\t<DataArray type=\"Float32\" Name=\"X\" "
         "NumberOfComponents=\"1\" format=\"ascii\">\n";
  ofs << "\t\t\t\t</DataArray>\n";
  for (size_t x = bounds.x; x < bounds.x + bounds.width; ++x) {
    ofs << x << " ";
  }
  ofs << "\n";
  ofs << "\t\t\t\t<DataArray type=\"Float32\" Name=\"Y\" "
         "NumberOfComponents=\"1\" format=\"ascii\">\n";
  ofs << "\t\t\t\t</DataArray>\n";
  for (size_t y = bounds.y; y < bounds.y + bounds.height; ++y) {
    ofs << y << " ";
  }
  ofs << "\n";
  ofs << "\t\t\t\t<DataArray type=\"Float32\" Name=\"Z\" "
         "NumberOfComponents=\"1\" format=\"ascii\">\n";
  for (size_t z = 0; z < particle.frameCount(); ++z) {
    ofs << particle.frame(z) << " ";
  }
  ofs << "\n";
  ofs << "\t\t\t\t</DataArray>\n";
  ofs << "\t\t\t</Coordinates>\n";
  ofs << "\t\t</Piece>\n";
  ofs << "\t</RectilinearGrid>\n";
  ofs << "</VTKFile>\n";
  return false;
}
