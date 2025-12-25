#ifndef MACRO_H
#define MACRO_H

#include <cstdio>
#include <cstdlib>

using namespace std;

#define CUDA_CHECK(err)                                                        \
  {                                                                            \
    if (err != 0) {                                                            \
      fprintf(stderr, "Error code: %d\n", err);                                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// SỬA: Tính toán chỉ số 3D chuẩn xác (Channel-First: Depth -> Height -> Width)
#define GET_1D_IDX(i, j, d, width, height)                                     \
  (((d) * (width) * (height)) + ((i) * (width)) + (j))

#define SQR(x) ((x) * (x))

#endif