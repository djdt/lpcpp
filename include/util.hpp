#pragma once

#include <chrono>

std::chrono::duration<double> get_remaining_time(
    std::chrono::time_point<std::chrono::system_clock> start_time, int n,
    int count, double &fps);

template <typename Iter, typename IdxIter>
auto remove_indices(Iter begin, Iter end, IdxIter indices_begin,
                    IdxIter indices_end) -> Iter {

  while (indices_end != indices_begin) {
    --indices_end;
    auto pos = begin + *indices_end;
    std::move(std::next(pos), end--, pos);
  }
  return end;
}
