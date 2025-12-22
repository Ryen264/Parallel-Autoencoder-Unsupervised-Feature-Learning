#include "cpu_utils.h"

std::string format_time(float time_ms) {
  std::stringstream builder;
  time_ms /= 1000;

  if (time_ms < 60) {
    builder << std::fixed << std::setprecision(3) << time_ms << 's';
    return builder.str();
  }

  auto [total_min, sec] = std::div(time_ms, 60);
  auto [hour, min]      = std::div(total_min, 60);

  int entry = 0;

  if (hour != 0) {
    builder << hour << 'h';
    ++entry;
  }

  if (min != 0) {
    if (entry)
      builder << ' ';
    builder << min << 'm';
    ++entry;
  }

  if (entry)
    builder << ' ';
  builder << sec << 's';

  return builder.str();
}