#ifndef MACRO_H
#define MACRO_H

#define CUDA_CHECK(err)                                                                \
  {                                                                                    \
    if (err != cudaSuccess) {                                                          \
      fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));                    \
      exit(EXIT_FAILURE);                                                              \
    }                                                                                  \
  }

#endif