#include "cpu_layers.h"

void cpu_conv2D(float *in, float *filter, float *out, int n, int width, int height, int depth, int n_filter) {
  for (int image = 0; image < n; ++image) {
    int   image_offset = image * width * height;
    float *in_offset    = in + image_offset * depth;
    float *out_offset   = out + image_offset * n_filter;

    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          for (int f = 0; f < n_filter; ++f) {
            float sum = 0;
            float *filter_offset = filter + f * CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT * depth;
            
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

                sum += in_offset[GET_1D_IDX(row, col, d, width, height)] *
                       filter_offset[GET_1D_IDX(f_i, f_j, d, CONV_FILTER_WIDTH, CONV_FILTER_HEIGHT)];
              }
            }
            out_offset[GET_1D_IDX(i, j, d, width, height)] = sum;
          }
        }
      }
    }
  }
}

void cpu_add_bias(float *in, float *bias, float *out, int n, int width, int height, int depth) {
  for (int image = 0; image < n; ++image) {
    for (int d = 0; d < depth; ++d) {
      int offset = image * width * height * depth + d * width * height;
      float bias_val = bias[d];
      float *in_offset = in + offset;
      float *out_offset = out + offset;

      for (int i = 0; i < width * height; ++i)
        out_offset[i] = in_offset[i] + bias_val;
    }
  }
}

void cpu_relu(float *in, float *out, int n, int width, int height, int depth) {
  for (int i = 0; i < n * width * height * depth; ++i)
    out[i] = max(0.0f, in[i]);
}

void cpu_avg_pooling(float *in, float *out, int n, int width, int height, int depth) {
  for (int image = 0; image < n; ++image) {
    int offset = image * width * height * depth;
    float *in_offset  = in + offset;
    float *out_offset = out + offset / 4;
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < height / 2; ++i) {
        for (int j = 0; j < width / 2; ++j) {
          // Get current out pixel index (depth 0)
          float sum = 0;
          // Get indices of filtered elements (depth 0)
          int neighbors_idx[] = {
            GET_1D_IDX(i * 2,      j * 2,      d, width, height),
            GET_1D_IDX(i * 2,      j * 2 + 1,  d, width, height),
            GET_1D_IDX(i * 2 + 1,  j * 2,      d, width, height),
            GET_1D_IDX(i * 2 + 1,  j * 2 + 1,  d, width, height),
          };

          // Apply for all depths
          for (int neighbor : neighbors_idx)
            sum += in_offset[neighbor];
          out_offset[GET_1D_IDX(i, j, d, width / 2, height / 2)] = sum / 4.0f;
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
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          // Get current in pixel index (depth 0)
          float val = in_offset[GET_1D_IDX(i, j, d, width, height)];
          // Get indices of filtered elements (depth 0)
          int neighbors_idx[] = {
            GET_1D_IDX(i * 2,     j * 2,      d, 2 * width, 2 * height),
            GET_1D_IDX(i * 2,     j * 2 + 1,  d, 2 * width, 2 * height),
            GET_1D_IDX(i * 2 + 1, j * 2,      d, 2 * width, 2 * height),
            GET_1D_IDX(i * 2 + 1, j * 2 + 1,  d, 2 * width, 2 * height),
          };

          // Apply for all depths
          for (int neighbor_idx : neighbors_idx)
            out_offset[neighbor_idx] = val;
        }
      }
    }
  }
}

float cpu_mse_loss(float *expected, float *actual, int n, int width, int height, int depth) {
  float sum   = 0;
  int   total = n * width * height * depth;
  for (int i = 0; i < total; ++i)
    sum += SQR(expected[i] - actual[i]);
  return sum / total;
}

void cpu_mse_grad(float *expected, float *actual, float *d_out, int n, int width, int height, int depth) {
  int total = n * width * height * depth;
  for (int i = 0; i < total; ++i)
    d_out[i] = 2.0f * (actual[i] - expected[i]) / total;
}

void cpu_relu_backward(float *in, float *d_out, float *d_in, int n, int width, int height, int depth) {
  for (int i = 0; i < n * width * height * depth; ++i)
    d_in[i] = in[i] > 0 ? d_out[i] : 0;
}

void cpu_avg_pooling_backward(float *d_out, float *d_in, int n, int width, int height, int depth) {
  for (int image = 0; image < n; ++image) {
    int offset = image * width * height * depth;
    float *d_in_offset  = d_in + offset;
    float *d_out_offset = d_out + offset / 4;
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          d_in_offset[GET_1D_IDX(i, j, d, width, height)] = d_out_offset[GET_1D_IDX(i / 2, j / 2, d, width / 2, height / 2)] / 4.0f;
        }
      }
    }
  }
}

void cpu_upsampling_backward(float *d_out, float *d_in, int n, int width, int height, int depth) {
  for (int image = 0; image < n; ++image) {
    int offset = image * width * height * depth;
    float *d_in_offset  = d_in + offset;
    float *d_out_offset = d_out + offset * 4;
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          // Get current in pixel index (depth 0)
          float sum = 0.0f;
          int neighbors_idx[] = {
            GET_1D_IDX(i * 2,     j * 2,      d, 2 * width, 2 * height),
            GET_1D_IDX(i * 2,     j * 2 + 1,  d, 2 * width, 2 * height),
            GET_1D_IDX(i * 2 + 1, j * 2,      d, 2 * width, 2 * height),
            GET_1D_IDX(i * 2 + 1, j * 2 + 1,  d, 2 * width, 2 * height)
          };

          for (int neighbor_idx : neighbors_idx)
            sum += d_out_offset[neighbor_idx];
          // Then add to d_in
          d_in_offset[GET_1D_IDX(i, j, d, width, height)] = sum;
        }
      }
    }
  }
}

void cpu_bias_grad(float *d_out, float *d_bias, int n, int width, int height, int depth) {
  for (int d = 0; d < depth; ++d) {
    // Sum of all d_out in the corresponding depth
    float sum = 0;
    for (int image = 0; image < n; ++image) {
      float *d_out_offset = d_out + d * width * height + image * width * height * depth;
      for (int i = 0; i < width * height; ++i)
        sum += d_out_offset[i];
    }
    d_bias[d] = sum;
  }
}

void cpu_conv2D_grad(float *in, float *d_out, float *d_filter, int n, int width, int height, int depth, int n_filter) {
  // Set all value of grad to 0
  memset(d_filter, 0, CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT * depth * n_filter * sizeof(float));

  // Calculate gradient
  for (int image = 0; image < n; ++image) {
    int image_offset = image * width * height;
    float *in_offset = in + image_offset * depth;
    float *d_out_offset = d_out + image_offset * n_filter;

    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          for (int f = 0; f < n_filter; ++f) {
            float *d_filter_offset = d_filter + f * CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT * depth;

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

                d_filter_offset[GET_1D_IDX(f_i, f_j, d, CONV_FILTER_WIDTH, CONV_FILTER_HEIGHT)] += in_offset[GET_1D_IDX(row, col, d, width, height)]
                                                                                                * d_out_offset[GET_1D_IDX(i, j, f, width, height)];
              }
            }
          }
        }
      }
    }
  }
}

void cpu_update_weight(float *weight, float *gradient, int size, float learning_rate) {
  for (int i = 0; i < size; ++i)
    weight[i] -= learning_rate * gradient[i];
}