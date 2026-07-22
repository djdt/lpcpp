#include <array>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

enum PreprocessImageMode {
  PROC_MODE_NORMAL, // light
  PROC_MODE_INVERT, // dark
  PROC_MODE_ABSOLUTE,
};

std::array<float, 3> find_capillary(cv::InputArray &input);

double image_center_weighted_intensity(cv::InputArray &image,
                                       cv::InputArray &mask,
                                       cv::OutputArray &weights);
double image_intensity(cv::InputArray &image, cv::InputArray &mask);
double image_sharpness(cv::InputArray &image, cv::OutputArray &laplace);

void unsharp_mask(cv::InputArray &image, cv::OutputArray &output, double alpha);

void update_background(cv::InputArray &frame, cv::InputOutputArray &mean,
                       cv::InputOutputArray &var, int pos);

bool init_background(cv::VideoCapture &cap, cv::InputOutputArray &mean,
                     cv::InputOutputArray &var, int frame_count);

void preprocess_and_threshold(cv::InputArray &frame, cv::InputArray &mean,
                              cv::InputArray &var, cv::OutputArray &processed,
                              cv::OutputArray &threshold,
                              const double zscore = 3.0,
                              const double unsharp_alpha = 1.0,
                              const PreprocessImageMode = PROC_MODE_INVERT);
