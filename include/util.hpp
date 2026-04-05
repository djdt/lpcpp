#pragma once

#include <chrono>

std::chrono::duration<double> get_remaining_time(
    std::chrono::time_point<std::chrono::system_clock> start_time, int n,
    int count, double &fps);
