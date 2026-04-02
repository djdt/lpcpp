// Macro for a 9-element sorting network optimized for the median
#include <opencv2/core/cuda.hpp>

#define CAS(a, b)                                                              \
  {                                                                            \
    float temp = a;                                                            \
    a = min(a, b);                                                             \
    b = max(temp, b);                                                          \
  }

__global__ void median3x3Kernel(const float *src, float *dst, int step,
                                int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Boundary check for 3x3 window
  if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
    float p[9];

    // Load 3x3 neighborhood into registers (direct global memory access)
    p[0] = src[(y - 1) * step + (x - 1)];
    p[1] = src[(y - 1) * step + x];
    p[2] = src[(y - 1) * step + (x + 1)];
    p[3] = src[y * step + (x - 1)];
    p[4] = src[y * step + x];
    p[5] = src[y * step + (x + 1)];
    p[6] = src[(y + 1) * step + (x - 1)];
    p[7] = src[(y + 1) * step + x];
    p[8] = src[(y + 1) * step + (x + 1)];

    // Sorting network for 9 elements
    CAS(p[1], p[2]);
    CAS(p[4], p[5]);
    CAS(p[7], p[8]);
    CAS(p[0], p[1]);
    CAS(p[3], p[4]);
    CAS(p[6], p[7]);
    CAS(p[1], p[2]);
    CAS(p[4], p[5]);
    CAS(p[7], p[8]);
    CAS(p[0], p[3]);
    CAS(p[5], p[8]);
    CAS(p[4], p[7]);
    CAS(p[3], p[6]);
    CAS(p[1], p[4]);
    CAS(p[2], p[5]);
    CAS(p[4], p[7]);
    CAS(p[4], p[2]);
    CAS(p[6], p[4]);
    CAS(p[4], p[2]);

    dst[y * step + x] = p[4]; // Resulting median is in the middle index
  }
}

void medianFilter3x3(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst) {
  dim3 block(32, 8);
  dim3 grid((src.cols + block.x - 1) / block.x,
            (src.rows + block.y - 1) / block.y);

  //// Convert step from bytes to number of float elements
  int step = src.step / sizeof(float);
  // Launch on the default (synchronous) stream
  median3x3Kernel<<<grid, block>>>(src.ptr<float>(), dst.ptr<float>(), step,
                                   src.cols, src.rows);

  // Ensure the GPU finishes before returning to CPU
  cudaDeviceSynchronize();
}
