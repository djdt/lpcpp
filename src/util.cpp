#include "util.hpp"

std::chrono::duration<double> get_remaining_time(
    const std::chrono::time_point<std::chrono::system_clock> &start_time,
    const std::chrono::time_point<std::chrono::system_clock> &current_time,
    const int n, const int count, double &fps) {
  auto duration = std::chrono::duration<double>(current_time - start_time);
  std::chrono::duration remaining =
      duration * (static_cast<double>(count) / static_cast<double>(n)) -
      duration;
  fps = static_cast<double>(n) / duration.count();
  return remaining;
}
