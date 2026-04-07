#pragma once

#include <condition_variable>
#include <opencv2/videoio.hpp>
#include <thread>

class AsyncVideoCapture {
private:
  cv::VideoCapture cap;
  cv::Mat frame;
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cv;
  std::atomic<bool> running;
  bool frame_ready;

public:
  AsyncVideoCapture(const std::string &filename, int api);

  ~AsyncVideoCapture();

  void read(cv::Mat &output);

  int get(const int prop);
  bool set(const int prop, const double value);

  void invalidate();

private:
  void update();
};
