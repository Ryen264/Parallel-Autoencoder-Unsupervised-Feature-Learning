#include "timer.h"

Timer::Timer() {
  cudaEventCreate(&_start);
  cudaEventCreate(&_stop);
}

Timer::~Timer() {
  cudaEventDestroy(_start);
  cudaEventDestroy(_stop);
}

void Timer::start() {
  cudaEventRecord(_start, 0);
  cudaEventSynchronize(_start);
}

void Timer::stop() { cudaEventRecord(_stop, 0); }

float Timer::get() {
  float elapsed;
  cudaEventSynchronize(_stop);
  cudaEventElapsedTime(&elapsed, _start, _stop);
  return elapsed;
}