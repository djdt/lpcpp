#pragma once

#include <chrono>

std::chrono::duration<double> get_remaining_time(
    const std::chrono::time_point<std::chrono::system_clock> &start_time,
    const std::chrono::time_point<std::chrono::system_clock> &current_time,
    const int n, const int count, double &fps);
