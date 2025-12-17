#include "utils.h"

string format_time(float time_ms) {
  stringstream builder;
  time_ms /= 1000;

  if (time_ms < 60) {
    builder << fixed << setprecision(3) << time_ms << 's';
    return builder.str();
  }

  int total_min = static_cast<int>(time_ms) / 60;
  int sec = static_cast<int>(time_ms) % 60;
  int hour = total_min / 60;
  int min = total_min % 60;

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