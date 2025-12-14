#ifndef TIMER_H
#define TIMER_H

/**
 * @brief Timer class
 *
 */
class Timer {
  cudaEvent_t _start;
  cudaEvent_t _stop;

public:
  GpuTimer();
  ~GpuTimer();

  void  start();
  void  stop();
  float get();
};

#endif