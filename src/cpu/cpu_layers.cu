#include "constants.h"
#include "cpu_layers.h"
#include <algorithm>
#include <cstring>

using std::max_element;
using std::memset;

#define SQR(x)                              ((x) * (x))
#define MAX(a, b)                           ((a) > (b) ? (a) : (b))
#define GET_1D_INDEX(i, j, k, width, depth) ((k) + (depth) * ((j) + (i) * (width)))

void cpu_conv2D(
    float *in, float *filter, float *out, int n, int width, int depth, int n_filter) {
  const int FILTER_DEPTH = CONV_FILTER_WIDTH * CONV_FILTER_WIDTH * depth * n_filter;

  for (int image = 0; image < n; ++image) {
    for (int i = 0; i < width; ++i) {
      for (int j = 0; j < width; ++j) {
        // Get current out pixel index (depth 0)
        float *cur = out + image * GET_1D_INDEX(i, j, 0, width, n_filter);

        for (int f = 0; f < n_filter; ++f) {
          float sum = 0;
          for (int f_i = 0; f_i < CONV_FILTER_WIDTH; ++f_i) {
            // If the row needs padding, we skip since we pad with 0
            int row = i + f_i - CONV_FILTER_WIDTH / 2;
            if (row < 0 || row >= width)
              continue;

            for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {
              // Same with column
              int col = j + f_j - CONV_FILTER_WIDTH / 2;
              if (col < 0 || col >= width)
                continue;

              // Calculate start of filter
              float *cur_filter =
                  filter + f * GET_1D_INDEX(f_i, f_j, 0, CONV_FILTER_WIDTH, depth);

              // Calculate start of input
              float *in_start = in + image * GET_1D_INDEX(row, col, 0, width, depth);

              for (int d = 0; d < depth; ++d) {
                sum += in_start[d] * cur_filter[d];
              }
            }
          }

          cur[f] = sum;
        }
      }
    }
  }
}

void cpu_add_bias(float *in, float *bias, float *out, int n, int width, int depth) {
  for (int i = 0; i < n * width * width * depth; i += depth) {
    float *in_offset  = in + depth;
    float *out_offset = out + depth;
    for (int d = 0; d < depth; ++d)
      out_offset[d] = in_offset[d] + bias[d];
  }
}

void cpu_relu(float *in, float *out, int n, int width, int depth) {
  for (int i = 0; i < n * width * width * depth; ++i)
    out[i] = MAX(0.0, in[i]);
}

void cpu_max_pooling(float *in, float *out, int n, int width, int depth) {
  for (int image = 0; image < n; ++image) {
    for (int i = 0; i < width / 2; ++i) {
      for (int j = 0; j < width / 2; ++j) {
        // Get current out pixel index (depth 0)
        float *cur = out + image * GET_1D_INDEX(i, j, 0, width / 2, depth);
        // Get indices of filtered elements (depth 0)
        int neighbors_idx[] = { image * GET_1D_INDEX(i * 2, j * 2, 0, width, depth),
                                image * GET_1D_INDEX(i * 2, j * 2 + 1, 0, width, depth),
                                image * GET_1D_INDEX(i * 2 + 1, j * 2, 0, width, depth),
                                image * GET_1D_INDEX(
                                            i * 2 + 1, j * 2 + 1, 0, width, depth) };

        // Apply for all depths
        for (int k = 0; k < depth; ++k) {
          float cur_val = in[neighbors_idx[0] + k];
          for (int neighbor = 1; neighbor < 4; ++neighbor)
            cur_val = MAX(cur_val, in[neighbors_idx[neighbor] + k]);
          cur[k] = cur_val;
        }
      }
    }
  }
}

void cpu_upsampling(float *in, float *out, int n, int width, int depth) {
  for (int image = 0; image < n; ++image) {
    for (int i = 0; i < width; ++i) {
      for (int j = 0; j < width; ++j) {
        // Get current in pixel index (depth 0)
        float *cur = in + image * GET_1D_INDEX(i, j, 0, width, depth);
        // Get indices of filtered elements (depth 0)
        int neighbors_idx[] = {
          image * GET_1D_INDEX(i * 2, j * 2, 0, 2 * width, depth),
          image * GET_1D_INDEX(i * 2, j * 2 + 1, 0, 2 * width, depth),
          image * GET_1D_INDEX(i * 2 + 1, j * 2, 0, 2 * width, depth),
          image * GET_1D_INDEX(i * 2 + 1, j * 2 + 1, 0, 2 * width, depth)
        };

        // Apply for all depths
        for (int k = 0; k < depth; ++k) {
          for (int neighbor_idx : neighbors_idx)
            out[neighbor_idx + k] = cur[k];
        }
      }
    }
  }
}

float cpu_mse_loss(float *expected, float *actual, int n, int width, int depth) {
  float sum   = 0;
  int   total = n * width * width * depth;
  for (int i = 0; i < total; ++i)
    sum += SQR(expected[i] - actual[i]);

  return sum / total;
}

void cpu_mse_grad(
    float *expected, float *actual, float *d_out, int n, int width, int depth) {
  int total = n * width * width * depth;
  for (int i = 0; i < total; ++i)
    d_out[i] = 2.0f * (actual[i] - expected[i]) / total;
}

void
cpu_relu_backward(float *in, float *d_out, float *d_in, int n, int width, int depth) {
  for (int i = 0; i < n * width * width * depth; ++i) {
    d_in[i] = in[i] > 0 ? d_out[i] : 0;
  }
}

void cpu_max_pooling_backward(
    float *in, float *d_out, float *d_in, int n, int width, int depth) {
  for (int image = 0; image < n; ++image) {
    for (int i = 0; i < width / 2; ++i) {
      for (int j = 0; j < width / 2; ++j) {
        // Get current d_out pixel index (depth 0)
        float *cur = d_out + image * GET_1D_INDEX(i, j, 0, width / 2, depth);
        // Get indices of filtered elements (depth 0)
        int neighbors_idx[] = { image * GET_1D_INDEX(i * 2, j * 2, 0, width, depth),
                                image * GET_1D_INDEX(i * 2, j * 2 + 1, 0, width, depth),
                                image * GET_1D_INDEX(i * 2 + 1, j * 2, 0, width, depth),
                                image * GET_1D_INDEX(
                                            i * 2 + 1, j * 2 + 1, 0, width, depth) };

        for (int k = 0; k < depth; ++k) {
          // Find argmax of the neighbors
          int max_neighbor_idx =
              *max_element(neighbors_idx, neighbors_idx + 4, [in, k](int a, int b) {
                return in[a + k] < in[b + k];
              });

          // Set d_in at argmax be d_out, else 0
          for (int neighbor_idx : neighbors_idx) {
            d_in[neighbor_idx + k] = (neighbor_idx == max_neighbor_idx) ? cur[k] : 0;
          }
        }
      }
    }
  }
}

void cpu_upsampling_backward(float *d_out, float *d_in, int n, int width, int depth) {
  for (int image = 0; image < n; ++image) {
    for (int i = 0; i < width; ++i) {
      for (int j = 0; j < width; ++j) {
        // Get current in pixel index (depth 0)
        float *cur = d_in + image * GET_1D_INDEX(i, j, 0, width, depth);
        // Get indices of filtered elements (depth 0)
        int neighbors_idx[] = {
          image * GET_1D_INDEX(i * 2, j * 2, 0, 2 * width, depth),
          image * GET_1D_INDEX(i * 2, j * 2 + 1, 0, 2 * width, depth),
          image * GET_1D_INDEX(i * 2 + 1, j * 2, 0, 2 * width, depth),
          image * GET_1D_INDEX(i * 2 + 1, j * 2 + 1, 0, 2 * width, depth)
        };

        for (int k = 0; k < depth; ++k) {
          // Sum of all deltas
          float sum = 0;
          for (int neighbor_idx : neighbors_idx)
            sum += d_out[neighbor_idx + k];

          // Then add to d_in
          cur[k] = sum;
        }
      }
    }
  }
}

void cpu_bias_grad(float *d_out, float *grad_bias, int n, int width, int depth) {
  for (int d = 0; d < depth; ++d) {
    // Sum of all d_out in the corresponding depth
    float *d_out_offset = d_out + d;
    float  sum          = 0;
    for (int i = 0; i < n * width * width * depth; i += depth)
      sum += d_out_offset[i];
    grad_bias[d] = sum;
  }
}

void cpu_conv2D_grad(float *in,
                     float *d_out,
                     float *grad_filter,
                     int    n,
                     int    width,
                     int    depth,
                     int    n_filter) {
  // Set all value of grad to 0
  memset(grad_filter,
         0,
         CONV_FILTER_WIDTH * CONV_FILTER_WIDTH * depth * n_filter * sizeof(float));

  // Calculate gradient
  for (int image = 0; image < n; ++image) {
    for (int i = 0; i < width; ++i) {
      for (int j = 0; j < width; ++j) {
        for (int f = 0; f < n_filter; ++f) {
          float d_out_val = d_out[image * GET_1D_INDEX(i, j, f, width, n_filter)];

          for (int f_i = 0; f_i < CONV_FILTER_WIDTH; ++f_i) {
            // If the row needs padding, we skip since we pad with 0
            int row = i + f_i - CONV_FILTER_WIDTH / 2;
            if (row < 0 || row >= width)
              continue;

            for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {
              // Same with column
              int col = j + f_j - CONV_FILTER_WIDTH / 2;
              if (col < 0 || col >= width)
                continue;

              // Calculate start of filter
              float *grad_filter_start =
                  grad_filter + f * GET_1D_INDEX(f_i, f_j, 0, CONV_FILTER_WIDTH, depth);

              // Calculate start of input
              float *in_start = in + image * GET_1D_INDEX(row, col, 0, width, depth);

              for (int d = 0; d < depth; ++d)
                grad_filter_start[d] += in_start[d] * d_out_val;
            }
          }
        }
      }
    }
  }
}

void cpu_update_weight(float *in, float *grad, int size, float learning_rate) {
  for (int i = 0; i < size; ++i)
    in[i] -= learning_rate * grad[i];
}