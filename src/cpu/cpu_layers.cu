#include "cpu/cpu_layers.h"

void cpu_conv2D(float *in,
                float *filter,
                float *out,
                int    n,
                int    width,
                int    height,
                int    depth,
                int    n_filter) {
  for (int image = 0; image < n; ++image) {
    int    image_offset = image * width * height;
    float *in_offset    = in + image_offset * depth;
    float *out_offset   = out + image_offset * n_filter;
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        // Get current out pixel index (depth 0)
        float *cur = out_offset + GET_1D_IDX(i, j, 0, width, n_filter);

        for (int f = 0; f < n_filter; ++f) {
          float  sum = 0;
          float *filter_offset =
              filter + f * CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT * depth;
          for (int f_i = 0; f_i < CONV_FILTER_HEIGHT; ++f_i) {
            // If the row needs padding, we skip since we pad with 0
            int row = i + f_i - CONV_FILTER_HEIGHT / 2;
            if (row < 0 || row >= height)
              continue;

            for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {
              // Same with column
              int col = j + f_j - CONV_FILTER_WIDTH / 2;
              if (col < 0 || col >= width)
                continue;

              // Calculate start of filter
              float *cur_filter =
                  filter_offset + GET_1D_IDX(f_i, f_j, 0, CONV_FILTER_WIDTH, depth);

              // Calculate start of input
              float *in_start = in_offset + GET_1D_IDX(row, col, 0, width, depth);

              for (int d = 0; d < depth; ++d)
                sum += in_start[d] * cur_filter[d];
            }
          }
          cur[f] = sum;
        }
      }
    }
  }
}

void cpu_add_bias(
    float *in, float *bias, float *out, int n, int width, int height, int depth) {
  for (int i = 0; i < n * width * height * depth; i += depth) {
    float *in_offset  = in + i;
    float *out_offset = out + i;
    for (int d = 0; d < depth; ++d)
      out_offset[d] = in_offset[d] + bias[d];
  }
}

void cpu_relu(float *in, float *out, int n, int width, int height, int depth) {
  for (int i = 0; i < n * width * height * depth; ++i)
    out[i] = max(0.0f, in[i]);
}

void cpu_avg_pooling(float *in, float *out, int n, int width, int height, int depth) {
  for (int image = 0; image < n; ++image) {
    int    offset     = image * width * height * depth;
    float *in_offset  = in + offset;
    float *out_offset = out + offset / 4;
    for (int i = 0; i < height / 2; ++i) {
      for (int j = 0; j < width / 2; ++j) {
        // Get current out pixel index (depth 0)
        float *cur = out_offset + GET_1D_IDX(i, j, 0, width / 2, depth);
        // Get indices of filtered elements (depth 0)
        int neighbors_idx[] = {
          GET_1D_IDX(i * 2, j * 2, 0, width, depth),
          GET_1D_IDX(i * 2, j * 2 + 1, 0, width, depth),
          GET_1D_IDX(i * 2 + 1, j * 2, 0, width, depth),
          GET_1D_IDX(i * 2 + 1, j * 2 + 1, 0, width, depth),
        };

        // Apply for all depths
        for (int k = 0; k < depth; ++k) {
          float sum = 0;
          for (int neighbor : neighbors_idx)
            sum += in_offset[neighbor + k];
          cur[k] = sum / 4.0f;
        }
      }
    }
  }
}

void cpu_upsampling(float *in, float *out, int n, int width, int height, int depth) {
  for (int image = 0; image < n; ++image) {
    int    offset     = image * width * height * depth;
    float *in_offset  = in + offset;
    float *out_offset = out + offset * 4;
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        // Get current in pixel index (depth 0)
        float *cur = in_offset + GET_1D_IDX(i, j, 0, width, depth);
        // Get indices of filtered elements (depth 0)
        int neighbors_idx[] = { GET_1D_IDX(i * 2, j * 2, 0, 2 * width, depth),
                                GET_1D_IDX(i * 2, j * 2 + 1, 0, 2 * width, depth),
                                GET_1D_IDX(i * 2 + 1, j * 2, 0, 2 * width, depth),
                                GET_1D_IDX(i * 2 + 1, j * 2 + 1, 0, 2 * width, depth) };

        // Apply for all depths
        for (int neighbor_idx : neighbors_idx) {
          for (int k = 0; k < depth; ++k)
            out_offset[neighbor_idx + k] = cur[k];
        }
      }
    }
  }
}

float
cpu_mse_loss(float *expected, float *actual, int n, int width, int height, int depth) {
  float sum   = 0;
  int   total = n * width * height * depth;
  for (int i = 0; i < total; ++i)
    sum += SQR(expected[i] - actual[i]);
  return sum / total;
}

void cpu_mse_grad(float *expected,
                  float *actual,
                  float *d_out,
                  int    n,
                  int    width,
                  int    height,
                  int    depth) {
  int total = n * width * height * depth;
  for (int i = 0; i < total; ++i)
    d_out[i] = 2.0f * (actual[i] - expected[i]) / total;
}

void cpu_relu_backward(
    float *in, float *d_out, float *d_in, int n, int width, int height, int depth) {
  for (int i = 0; i < n * width * height * depth; ++i)
    d_in[i] = in[i] > 0 ? d_out[i] : 0;
}

void cpu_avg_pooling_backward(
    float *d_out, float *d_in, int n, int width, int height, int depth) {
  for (int image = 0; image < n; ++image) {
    int    offset       = image * width * height * depth;
    float *d_in_offset  = d_in + offset;
    float *d_out_offset = d_out + offset / 4;
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        // Get current d_out pixel index (depth 0)
        float *cur    = d_out_offset + GET_1D_IDX(i, j, 0, width / 2, depth);
        float *target = d_in_offset + GET_1D_IDX(i, j, 0, width, depth);

        for (int k = 0; k < depth; ++k)
          target[k] = cur[k] / 4.0f;
      }
    }
  }
}

void cpu_upsampling_backward(
    float *d_out, float *d_in, int n, int width, int height, int depth) {
  for (int image = 0; image < n; ++image) {
    int    offset       = image * width * height * depth;
    float *d_in_offset  = d_in + offset;
    float *d_out_offset = d_out + offset * 4;
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        // Get current in pixel index (depth 0)
        float *cur = d_in_offset + GET_1D_IDX(i, j, 0, width, depth);
        // Get indices of filtered elements (depth 0)
        int neighbors_idx[] = { GET_1D_IDX(i * 2, j * 2, 0, 2 * width, depth),
                                GET_1D_IDX(i * 2, j * 2 + 1, 0, 2 * width, depth),
                                GET_1D_IDX(i * 2 + 1, j * 2, 0, 2 * width, depth),
                                GET_1D_IDX(i * 2 + 1, j * 2 + 1, 0, 2 * width, depth) };

        for (int k = 0; k < depth; ++k) {
          // Sum of all deltas
          float sum = 0;
          for (int neighbor_idx : neighbors_idx)
            sum += d_out_offset[neighbor_idx + k];

          // Then add to d_in
          cur[k] = sum;
        }
      }
    }
  }
}

void
cpu_bias_grad(float *d_out, float *grad_bias, int n, int width, int height, int depth) {
  for (int d = 0; d < depth; ++d) {
    // Sum of all d_out in the corresponding depth
    float *d_out_offset = d_out + d;
    float  sum          = 0;
    for (int i = 0; i < n * width * height * depth; i += depth)
      sum += d_out_offset[i];
    grad_bias[d] = sum;
  }
}

void cpu_conv2D_grad(float *in,
                     float *d_out,
                     float *grad_filter,
                     int    n,
                     int    width,
                     int    height,
                     int    depth,
                     int    n_filter) {
  // Set all value of grad to 0
  memset(grad_filter,
         0,
         CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT * depth * n_filter * sizeof(float));

  // Calculate gradient
  for (int image = 0; image < n; ++image) {
    int    image_offset = image * width * height;
    float *in_offset    = in + image_offset * depth;
    float *d_out_offset = d_out + image_offset * n_filter;

    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        for (int f = 0; f < n_filter; ++f) {
          float  d_out_val = d_out_offset[GET_1D_IDX(i, j, f, width, n_filter)];
          float *d_filter_offset =
              grad_filter + f * CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT * depth;

          for (int f_i = 0; f_i < CONV_FILTER_HEIGHT; ++f_i) {
            // If the row needs padding, we skip since we pad with 0
            int row = i + f_i - CONV_FILTER_HEIGHT / 2;
            if (row < 0 || row >= height)
              continue;

            for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {
              // Same with column
              int col = j + f_j - CONV_FILTER_WIDTH / 2;
              if (col < 0 || col >= width)
                continue;

              // Calculate start of filter
              float *grad_filter_start =
                  d_filter_offset + GET_1D_IDX(f_i, f_j, 0, CONV_FILTER_WIDTH, depth);

              // Calculate start of input
              float *in_start = in_offset + GET_1D_IDX(row, col, 0, width, depth);
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