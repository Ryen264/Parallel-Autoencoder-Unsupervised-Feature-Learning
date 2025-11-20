#include "cpu_layers.h"

#define SQR(x)                          ((x) * (x))
#define MAX(a, b)                       ((a) > (b) ? (a) : (b))
#define GET_1D_INDEX(i, j, k, n, depth) ((k) + (depth) * ((j) + (i) * (n)))

void cpu_conv2D(float *in, float *filter, float *out, int n, int depth, int n_filter) {
  const int FILTER_DEPTH = CONV_FILTER_SIZE * CONV_FILTER_SIZE * depth * n_filter;

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      // Get current out pixel index (depth 0)
      float *cur = out + GET_1D_INDEX(i, j, 0, n, n_filter);
      // Get filter offset
      int filter_offset = GET_1D_INDEX(i, j, 0, n, FILTER_DEPTH);

      for (int f = 0; f < n_filter; ++f) {
        float sum = 0;
        for (int f_i = 0; f_i < CONV_FILTER_SIZE; ++f_i) {
          // If the row needs padding, we skip since we pad with 0
          int row = i + f_i - CONV_FILTER_SIZE / 2;
          if (row < 0 || row >= n)
            continue;

          for (int f_j = 0; f_j < CONV_FILTER_SIZE; ++f_j) {
            // Same with column
            int col = j + f_j - CONV_FILTER_SIZE / 2;
            if (col < 0 || col >= n)
              continue;

            // Calculate start of filter
            float *cur_filter = filter +
                                filter_offset +
                                f *
                                GET_1D_INDEX(f_i, f_j, 0, CONV_FILTER_SIZE, depth);

            for (int d = 0; d < depth; ++d) {
              sum += in[GET_1D_INDEX(row, col, d, n, depth)] * cur_filter[d];
            }
          }
        }

        cur[f] = sum;
      }
    }
  }
}

void cpu_relu(float *in, float *out, int n, int depth) {
  for (int i = 0; i < n * n * depth; ++i)
    out[i] = MAX(0.0, in[i]);
}

void cpu_max_pooling(float *in, float *out, int n, int depth) {
  for (int i = 0; i < n; i += 2) {
    for (int j = 0; j < n; j += 2) {
      // Get current out pixel index (depth 0)
      float *cur = out + GET_1D_INDEX(i, j, 0, n / 2, depth);
      // Get indices of filtered elements (depth 0)
      int neighbors_idx[] = { GET_1D_INDEX(i * 2, j * 2, 0, n, depth),
                              GET_1D_INDEX(i * 2, j * 2 + 1, 0, n, depth),
                              GET_1D_INDEX(i * 2 + 1, j * 2, 0, n, depth),
                              GET_1D_INDEX(i * 2 + 1, j * 2 + 1, 0, n, depth) };

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

void cpu_upsampling(float *in, float *out, int n, int depth) {
  for (int i = 0; i < n; i += 2) {
    for (int j = 0; j < n; j += 2) {
      // Get current in pixel index (depth 0)
      float *cur = in + GET_1D_INDEX(i, j, 0, n / 2, depth);
      // Get indices of filtered elements (depth 0)
      int neighbors_idx[] = { GET_1D_INDEX(i * 2, j * 2, 0, n, depth),
                              GET_1D_INDEX(i * 2, j * 2 + 1, 0, n, depth),
                              GET_1D_INDEX(i * 2 + 1, j * 2, 0, n, depth),
                              GET_1D_INDEX(i * 2 + 1, j * 2 + 1, 0, n, depth) };

      // Apply for all depths
      for (int k = 0; k < depth; ++k) {
        for (int neighbor_idx : neighbors_idx)
          out[neighbor_idx + k] = cur[k];
      }
    }
  }
}

float cpu_mse_loss(float *expected, float *actual, int n, int depth) {
  int   vector_count = n * n;
  float sum          = 0;

  for (int i = 0; i < vector_count * depth; ++i)
    sum += SQR(expected[i] - actual[i]);

  return sum / vector_count;
}
