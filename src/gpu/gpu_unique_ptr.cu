#include "gpu_unique_ptr.h"
#include "macro.h"

Gpu_Unique_Ptr::Gpu_Unique_Ptr()
    : _ptr(0) {};

Gpu_Unique_Ptr::Gpu_Unique_Ptr(unsigned int n) {
  CUDA_CHECK(cudaMalloc(&_ptr, n * sizeof(float)));
}

Gpu_Unique_Ptr::Gpu_Unique_Ptr(Gpu_Unique_Ptr &&other) {
  _ptr       = other._ptr;
  other._ptr = 0;
}

Gpu_Unique_Ptr &Gpu_Unique_Ptr::operator=(Gpu_Unique_Ptr &&other) {
  _ptr       = other._ptr;
  other._ptr = 0;

  return *this;
}

Gpu_Unique_Ptr::~Gpu_Unique_Ptr() { reset(); }

float *Gpu_Unique_Ptr::get() const noexcept { return _ptr; }

void Gpu_Unique_Ptr::reset() {
  if (_ptr) {
    CUDA_CHECK(cudaFree(_ptr));
    _ptr = 0;
  }
}

Pinned_Unique_Ptr::Pinned_Unique_Ptr()
    : _ptr(0) {};

Pinned_Unique_Ptr::Pinned_Unique_Ptr(unsigned int n) {
  CUDA_CHECK(cudaMallocHost(&_ptr, n * sizeof(float)));
}

Pinned_Unique_Ptr::Pinned_Unique_Ptr(Pinned_Unique_Ptr &&other) {
  _ptr       = other._ptr;
  other._ptr = 0;
}

Pinned_Unique_Ptr &Pinned_Unique_Ptr::operator=(Pinned_Unique_Ptr &&other) {
  _ptr       = other._ptr;
  other._ptr = 0;

  return *this;
}

Pinned_Unique_Ptr::~Pinned_Unique_Ptr() { reset(); }

float *Pinned_Unique_Ptr::get() const noexcept { return _ptr; }

void Pinned_Unique_Ptr::reset() {
  if (_ptr) {
    CUDA_CHECK(cudaFreeHost(_ptr));
    _ptr = 0;
  }
}