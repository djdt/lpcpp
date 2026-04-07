#include <condition_variable>
#include <opencv2/videoio.hpp>
#include <thread>

#include "asynccapture.hpp"

AsyncVideoCapture::AsyncVideoCapture(const std::string &filename, int api)
    : frame_ready(false), running(true) {
  cap.open(filename, api);

  thread = std::thread(&AsyncVideoCapture::update, this);
}

AsyncVideoCapture::~AsyncVideoCapture() {
  running = false;
  cv.notify_all();
  if (thread.joinable())
    thread.join();
}

void AsyncVideoCapture::read(cv::Mat &output) {
  std::unique_lock<std::mutex> lock(mutex);

  cv.wait(lock, [this] { return frame_ready || !running.load(); });

  frame.copyTo(output);
  frame_ready = false;

  lock.unlock();
  cv.notify_one(); // wake up thread
}

int AsyncVideoCapture::get(const int prop) {
  std::lock_guard<std::mutex> lock(mutex);
  return cap.get(prop);
}

bool AsyncVideoCapture::set(const int prop, const double value) {
  std::lock_guard<std::mutex> lock(mutex);
  return cap.set(prop, value);
}

void AsyncVideoCapture::invalidate() {
  std::unique_lock<std::mutex> lock(mutex);
  frame_ready = false;
  lock.unlock();
  cv.notify_one();
}

void AsyncVideoCapture::update() {
  while (running) {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this] { return !frame_ready || !running.load(); });
    if (!running)
      break;

    cap.read(frame);
    frame_ready = true;

    lock.unlock();
    cv.notify_one(); // frame is ready
  }
}
