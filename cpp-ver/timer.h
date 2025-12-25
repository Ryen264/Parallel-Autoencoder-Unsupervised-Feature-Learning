#pragma once

#include <chrono>

class Timer {
private:
  // Sử dụng time_point để lưu trữ điểm thời gian bắt đầu và kết thúc
  std::chrono::time_point<std::chrono::high_resolution_clock> _start;
  std::chrono::time_point<std::chrono::high_resolution_clock> _stop;

public:
  // Constructor và Destructor (không cần làm gì với std::chrono)
  Timer();
  ~Timer();

  // Bắt đầu đo thời gian
  void start();
  // Dừng đo thời gian
  void stop();
  // Trả về thời gian đã trôi qua (elapsed time) bằng mili giây (ms)
  float get(); 
};