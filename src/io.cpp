#include "io.hpp"

#include "contours.hpp"
#include "cpuproc.hpp"
#include "particle.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

#ifdef ENABLE_HDF5_EXPORT
#include <H5Cpp.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/geometry.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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
         "circularity,convexity,intensity,maximum_feret,minimum_feret,maximum_"
         "legendre,minimum_legendre,mean_diameter,perimeter,sharpness,x,y"
      << std::endl;
}

void write_particle_data(const std::vector<Particle> &particles,
                         std::ofstream &ofs) {
  cv::Mat mask;
  for (auto it = particles.begin(); it != particles.end(); ++it) {
    const std::vector<cv::Point> &contour = it->contour();
    cv::Moments moments = cv::moments(contour);

    mask_for_contour(contour, mask);

    cv::Point center =
        cv::Point(moments.m10 / moments.m00, moments.m01 / moments.m00);

    cv::Point2f legendre = legendre_axes_from_moments(moments);

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
    ofs << legendre.y << "," << legendre.x << ",";
    ofs << contour_mean_distance(contour, center);
    ofs << cv::arcLength(contour, true) << ",";
    ofs << image_sharpness(it->image(), mask) << ",";
    ofs << center.x << "," << center.y << "\n";
  }
  ofs.flush();
}

bool save_particle_contours(const Particle &particle,
                            const std::filesystem::path &path) {
  std::ofstream ofs(path);
  for (int z = 0; z < particle.frameCount(); ++z) {
    const auto &contour = particle.contour(z);
    for (const auto &p : contour) {
      ofs << p.x << " " << p.y << " ";
    }
    ofs << "\n";
  }
  return false;
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

bool save_particle_data_vtk(const Particle &particle,
                            const std::filesystem::path &path) {
  cv::Rect bounds = particle.boundingRect();
  std::ofstream ofs(path);
  ofs << "<VTKFile type=\"ImageData\" version=\"0.1\" "
         "byte_order=\"LittleEndian\">\n";
  ofs << "\t<ImageData WholeExtent=\"0 " << bounds.width << " 0 "
      << bounds.height << " 0 " << particle.frameCount() << "\" Origin=\""
      << bounds.x << " " << bounds.y << " 0\" Spacing=\"1 1 1\">\n";
  ofs << "\t\t<Piece Extent=\"0 " << bounds.width << " 0 " << bounds.height
      << " 0 " << particle.frameCount() << "\">\n";

  ofs << "\t\t\t<CellData Scalars=\"Processed\">\n";
  ofs << "\t\t\t\t<DataArray type=\"Float32\" Name=\"Processed\" "
         "format=\"ascii\">\n";

  for (size_t z = 0; z < particle.frameCount(); ++z) {
    const cv::Mat &image = particle.image(z);
    cv::Rect rect = cv::boundingRect(particle.contour(z));
    cv::Point offset = rect.tl() - bounds.tl();

    for (size_t y = 0; y < bounds.height; ++y) {
      for (size_t x = 0; x < bounds.width; ++x) {
        size_t sx = x - offset.x;
        size_t sy = y - offset.y;
        if (sx >= 0 && sx < image.cols && sy >= 0 && sy < image.rows) {
          ofs << image.at<float>(sy, sx);
        } else {
          ofs << "0";
        }
        ofs << " ";
      }
    }
  }

  ofs << "\n\t\t\t\t</DataArray>\n";
  ofs << "\t\t\t\t<DataArray type=\"UInt8\" Name=\"Mask\" "
         "format=\"ascii\">\n";

  for (size_t z = 0; z < particle.frameCount(); ++z) {
    cv::Mat mask;
    cv::Rect rect = cv::boundingRect(particle.contour(z));
    cv::Point offset = rect.tl() - bounds.tl();
    mask_for_contour(particle.contour(z), mask);

    for (size_t y = 0; y < bounds.height; ++y) {
      for (size_t x = 0; x < bounds.width; ++x) {
        size_t sx = x - offset.x;
        size_t sy = y - offset.y;
        if (sx >= 0 && sx < mask.cols && sy >= 0 && sy < mask.rows) {
          ofs << static_cast<int>(mask.at<uchar>(sy, sx));
        } else {
          ofs << "0";
        }
        ofs << " ";
      }
    }
  }

  ofs << "\n\t\t\t\t</DataArray>\n";
  ofs << "\t\t\t</CellData>\n";

  ofs << "\t\t</Piece>\n";
  ofs << "\t</ImageData>\n";
  ofs << "</VTKFile>\n";
  return false;
}

#ifdef ENABLE_HDF5_EXPORT
bool save_particle_data_hdf5(const Particle &particle,
                             const std::filesystem::path &path) {
  cv::Rect bounds = particle.boundingRect();

  std::vector<float> buf;
  buf.reserve(bounds.height * bounds.width * particle.frameCount());
  std::vector<uchar> mbuf;
  mbuf.reserve(bounds.height * bounds.width * particle.frameCount());

  for (size_t z = 0; z < particle.frameCount(); ++z) {
    const cv::Mat &image = particle.image(z);
    cv::Mat mask;
    mask_for_contour(particle.contour(z), mask);

    cv::Rect rect = cv::boundingRect(particle.contour(z));
    cv::Point offset = rect.tl() - bounds.tl();

    for (size_t y = 0; y < bounds.height; ++y) {
      for (size_t x = 0; x < bounds.width; ++x) {
        int sx = x - offset.x;
        int sy = y - offset.y;
        if (sx >= 0 && sx < image.cols && sy >= 0 && sy < image.rows) {
          buf.push_back(image.at<float>(sy, sx));
          mbuf.push_back(mask.at<uchar>(sy, sx));
        } else {
          buf.push_back(0.f);
          mbuf.push_back(0);
        }
      }
    }
  }

  H5::H5File file(path, H5F_ACC_TRUNC);
  H5::Group root = file.createGroup("VTKHDF");

  int version[2] = {2, 3};

  root.createAttribute("Version", H5::PredType::NATIVE_INT,
                       H5::DataSpace(1, (hsize_t[]){2}))
      .write(H5::PredType::NATIVE_INT, version);

  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  root.createAttribute("Type", str_type, H5::DataSpace(H5S_SCALAR))
      .write(str_type, std::string("ImageData"));

  int extent[6] = {0, bounds.width, 0, bounds.height, 0, particle.frameCount()};
  int origin[3] = {bounds.x, bounds.y, particle.frame(0)};
  int spacing[3] = {1, 1, 1};
  int direction[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  root.createAttribute("WholeExtent", H5::PredType::NATIVE_INT,
                       H5::DataSpace(1, (hsize_t[]){6}))
      .write(H5::PredType::NATIVE_INT, extent);
  root.createAttribute("Origin", H5::PredType::NATIVE_INT,
                       H5::DataSpace(1, (hsize_t[]){3}))
      .write(H5::PredType::NATIVE_INT, origin);
  root.createAttribute("Spacing", H5::PredType::NATIVE_INT,
                       H5::DataSpace(1, (hsize_t[]){3}))
      .write(H5::PredType::NATIVE_INT, spacing);
  root.createAttribute("Direction", H5::PredType::NATIVE_INT,
                       H5::DataSpace(1, (hsize_t[]){9}))
      .write(H5::PredType::NATIVE_INT, direction);

  hsize_t dims[3] = {static_cast<hsize_t>(particle.frameCount()),
                     static_cast<hsize_t>(bounds.height),
                     static_cast<hsize_t>(bounds.width)};

  H5::DSetCreatPropList props;
  props.setChunk(3, dims);
  props.setDeflate(1);

  H5::Group point_data = root.createGroup("CellData");
  H5::DataSet proc = point_data.createDataSet(
      "Processed", H5::PredType::NATIVE_FLOAT, H5::DataSpace(3, dims), props);

  H5::DataSet mask = point_data.createDataSet(
      "Mask", H5::PredType::NATIVE_UCHAR, H5::DataSpace(3, dims), props);

  proc.write(buf.data(), H5::PredType::NATIVE_FLOAT);
  mask.write(mbuf.data(), H5::PredType::NATIVE_UCHAR);
  return false;
}
#endif
