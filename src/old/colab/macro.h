#ifndef MACRO_H
#define MACRO_H

#include <cstdio>
using namespace std;

#define CUDA_CHECK(err)                                                                \
  {                                                                                    \
    if (err != cudaSuccess) {                                                          \
      fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));                    \
      exit(EXIT_FAILURE);                                                              \
    }                                                                                  \
  }

#define GET_1D_IDX(i, j, d, width, depth)                                              \
  ((d) + (j) * (depth) + (i) * (width) * (depth))

#define SQR(x) ((x) * (x))

#endif