#include "timer.h"
#include <chrono>

using namespace std;

// Constructor
Timer::Timer() {
  // Không cần gọi hàm khởi tạo nào, std::chrono::time_point đã sẵn sàng
}

// Destructor
Timer::~Timer() {
  // Không cần gọi hàm giải phóng tài nguyên nào
}

void Timer::start() {
  // Ghi lại thời điểm hiện tại làm điểm bắt đầu
  _start = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
  // Ghi lại thời điểm hiện tại làm điểm dừng
  _stop = std::chrono::high_resolution_clock::now();
}

float Timer::get() {
  // Tính toán khoảng thời gian trôi qua giữa _stop và _start
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(_stop - _start);
  
  // Trả về giá trị dưới dạng float (mili giây)
  return (float)duration.count();
}