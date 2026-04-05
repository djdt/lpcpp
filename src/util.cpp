#include <chrono>

std::chrono::duration<double> get_remaining_time(
    std::chrono::time_point<std::chrono::system_clock> start_time, int n,
    int count, double &fps) {
  auto frame_time = std::chrono::system_clock::now();
  auto duration = std::chrono::duration<double>(frame_time - start_time);
  std::chrono::duration remaining =
      duration * (static_cast<double>(count) / static_cast<double>(n)) -
      duration;
  fps = static_cast<double>(n) / duration.count();
  return remaining;
}
