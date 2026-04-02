#pragma once

#include <opencv2/core/cuda.hpp>

void medianFilter3x3(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst);
