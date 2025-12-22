#include "cpu_layers.h"

// 1. CONVOLUTION FORWARD
void cpu_conv2D(float *in, float *filter, float *out, int n, int width, int height, int depth, int n_filter) {
  // 1. Reset toàn bộ output về 0 trước (vì ta sẽ cộng dồn)
  // Logic này tách biệt với bias (bias được cộng ở hàm cpu_add_bias riêng)
  memset(out, 0, n * width * height * n_filter * sizeof(float));

  for (int image = 0; image < n; ++image) {
    int image_offset = image * width * height;
    float *in_offset = in + image_offset * depth;
    float *out_offset = out + image_offset * n_filter;

    // Thứ tự vòng lặp chuẩn: Output Filter -> Height -> Width -> Input Depth
    for (int f = 0; f < n_filter; ++f) { // Output Channels
      for (int i = 0; i < height; ++i) { // Output Height
        for (int j = 0; j < width; ++j) { // Output Width
          
          float sum = 0; // Biến tích lũy cho 1 pixel (i, j, f)
          
          // Duyệt qua tất cả Input Channels để cộng dồn
          for (int d = 0; d < depth; ++d) { 
            
            float *filter_offset = filter + f * CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT * depth;
            
            for (int f_i = 0; f_i < CONV_FILTER_HEIGHT; ++f_i) {
              int row = i + f_i - CONV_FILTER_HEIGHT / 2;
              if (row < 0 || row >= height) continue;

              for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {
                int col = j + f_j - CONV_FILTER_WIDTH / 2;
                if (col < 0 || col >= width) continue;

                // Cộng dồn: sum += input * weight
                sum += in_offset[GET_1D_IDX(row, col, d, width, height)] *
                       filter_offset[GET_1D_IDX(f_i, f_j, d, CONV_FILTER_WIDTH, CONV_FILTER_HEIGHT)];
              }
            }
          }
          // Gán kết quả tổng hợp của tất cả input channels vào output
          out_offset[GET_1D_IDX(i, j, f, width, height)] = sum;
        }
      }
    }
  }
}

// 2. ADD BIAS
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

// 3. RELU FORWARD
void cpu_relu(float *in, float *out, int n, int width, int height, int depth) {
  for (int i = 0; i < n * width * height * depth; ++i)
    out[i] = max(0.0f, in[i]);
}

// 4. MAX POOLING FORWARD (Thay thế Avg Pooling)
// Logic: Tìm giá trị lớn nhất trong vùng 2x2
void cpu_max_pooling(float *in, float *out, int n, int width, int height, int depth) {
  for (int image = 0; image < n; ++image) {
    int offset = image * width * height * depth;
    float *in_offset  = in + offset;
    float *out_offset = out + offset / 4;
    
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < height / 2; ++i) {
        for (int j = 0; j < width / 2; ++j) {
          
          float max_val = -FLT_MAX; // Khởi tạo giá trị rất nhỏ

          int neighbors_idx[] = {
            GET_1D_IDX(i * 2,      j * 2,      d, width, height),
            GET_1D_IDX(i * 2,      j * 2 + 1,  d, width, height),
            GET_1D_IDX(i * 2 + 1,  j * 2,      d, width, height),
            GET_1D_IDX(i * 2 + 1,  j * 2 + 1,  d, width, height),
          };

          for (int neighbor : neighbors_idx) {
             if (in_offset[neighbor] > max_val) {
                 max_val = in_offset[neighbor];
             }
          }
          out_offset[GET_1D_IDX(i, j, d, width / 2, height / 2)] = max_val;
        }
      }
    }
  }
}

// 5. UPSAMPLING FORWARD
void cpu_upsampling(float *in, float *out, int n, int width, int height, int depth) {
  for (int image = 0; image < n; ++image) {
    int    offset     = image * width * height * depth;
    float *in_offset  = in + offset;
    float *out_offset = out + offset * 4;
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          float val = in_offset[GET_1D_IDX(i, j, d, width, height)];
          int neighbors_idx[] = {
            GET_1D_IDX(i * 2,     j * 2,      d, 2 * width, 2 * height),
            GET_1D_IDX(i * 2,     j * 2 + 1,  d, 2 * width, 2 * height),
            GET_1D_IDX(i * 2 + 1, j * 2,      d, 2 * width, 2 * height),
            GET_1D_IDX(i * 2 + 1, j * 2 + 1,  d, 2 * width, 2 * height),
          };
          for (int neighbor_idx : neighbors_idx)
            out_offset[neighbor_idx] = val;
        }
      }
    }
  }
}

// 6. MSE LOSS
float cpu_mse_loss(float *expected, float *actual, int n, int width, int height, int depth) {
  float sum   = 0;
  int   total = n * width * height * depth;
  for (int i = 0; i < total; ++i)
    sum += SQR(expected[i] - actual[i]);
  return sum / total;
}

// 7. MSE GRAD
void cpu_mse_grad(float *expected, float *actual, float *d_out, int n, int width, int height, int depth) {
  int total = n * width * height * depth;
  for (int i = 0; i < total; ++i)
    d_out[i] = 2.0f * (actual[i] - expected[i]) / total;
}

// 8. RELU BACKWARD
void cpu_relu_backward(float *in, float *d_out, float *d_in, int n, int width, int height, int depth) {
  for (int i = 0; i < n * width * height * depth; ++i)
    d_in[i] = in[i] > 0 ? d_out[i] : 0;
}

// 9. MAX POOLING BACKWARD (Thay thế Avg Pooling Backward)
// Lưu ý: Cần thêm tham số 'in' (ảnh đầu vào) để xác định lại vị trí max
void cpu_max_pooling_backward(float *in, float *d_out, float *d_in, int n, int width, int height, int depth) {
  // Reset d_in về 0 trước khi tính toán
  memset(d_in, 0, n * width * height * depth * sizeof(float));

  for (int image = 0; image < n; ++image) {
    int offset = image * width * height * depth;
    float *in_offset    = in + offset;       // Cần input gốc để tìm lại max
    float *d_in_offset  = d_in + offset;
    float *d_out_offset = d_out + offset / 4;

    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < height / 2; ++i) {
        for (int j = 0; j < width / 2; ++j) {
          
          // 1. Tìm lại vị trí Max trong vùng 2x2
          float max_val = -FLT_MAX;
          int max_idx = -1;

          int neighbors_idx[] = {
            GET_1D_IDX(i * 2,      j * 2,      d, width, height),
            GET_1D_IDX(i * 2,      j * 2 + 1,  d, width, height),
            GET_1D_IDX(i * 2 + 1,  j * 2,      d, width, height),
            GET_1D_IDX(i * 2 + 1,  j * 2 + 1,  d, width, height),
          };

          for (int idx : neighbors_idx) {
            if (in_offset[idx] > max_val) {
              max_val = in_offset[idx];
              max_idx = idx;
            }
          }

          // 2. Chỉ truyền gradient về đúng vị trí Max đó
          if (max_idx != -1) {
            float grad = d_out_offset[GET_1D_IDX(i, j, d, width / 2, height / 2)];
            d_in_offset[max_idx] += grad;
          }
        }
      }
    }
  }
}

// 10. UPSAMPLING BACKWARD
void cpu_upsampling_backward(float *d_out, float *d_in, int n, int width, int height, int depth) {
  for (int image = 0; image < n; ++image) {
    int offset = image * width * height * depth;
    float *d_in_offset  = d_in + offset;
    float *d_out_offset = d_out + offset * 4;
    for (int d = 0; d < depth; ++d) {
      for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
          float sum = 0.0f;
          int neighbors_idx[] = {
            GET_1D_IDX(i * 2,     j * 2,      d, 2 * width, 2 * height),
            GET_1D_IDX(i * 2,     j * 2 + 1,  d, 2 * width, 2 * height),
            GET_1D_IDX(i * 2 + 1, j * 2,      d, 2 * width, 2 * height),
            GET_1D_IDX(i * 2 + 1, j * 2 + 1,  d, 2 * width, 2 * height)
          };
          for (int neighbor_idx : neighbors_idx)
            sum += d_out_offset[neighbor_idx];
          d_in_offset[GET_1D_IDX(i, j, d, width, height)] = sum;
        }
      }
    }
  }
}

// 11. BIAS GRAD
void cpu_bias_grad(float *d_out, float *d_bias, int n, int width, int height, int depth) {
  for (int d = 0; d < depth; ++d) {
    float sum = 0;
    for (int image = 0; image < n; ++image) {
      float *d_out_offset = d_out + d * width * height + image * width * height * depth;
      for (int i = 0; i < width * height; ++i)
        sum += d_out_offset[i];
    }
    d_bias[d] = sum;
  }
}

// 12. CONV2D GRAD (TÍNH CHO FILTER)
void cpu_conv2D_grad(float *in, float *d_out, float *d_filter, int n, int width, int height, int depth, int n_filter) {
  memset(d_filter, 0, CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT * depth * n_filter * sizeof(float));

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
              int row = i + f_i - CONV_FILTER_HEIGHT / 2;
              if (row < 0 || row >= height) continue;

              for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {
                int col = j + f_j - CONV_FILTER_WIDTH / 2;
                if (col < 0 || col >= width) continue;

                d_filter_offset[GET_1D_IDX(f_i, f_j, d, CONV_FILTER_WIDTH, CONV_FILTER_HEIGHT)] += 
                    in_offset[GET_1D_IDX(row, col, d, width, height)] * d_out_offset[GET_1D_IDX(i, j, f, width, height)];
              }
            }
          }
        }
      }
    }
  }
}

// 13. CONV2D BACKWARD INPUT (TÍNH CHO D_IN) - Mới thêm vào
void cpu_conv2D_backward_input(float *d_out, float *filter, float *d_in, 
                               int n, int width, int height, int depth, int n_filter) {
  memset(d_in, 0, n * width * height * depth * sizeof(float));

  for (int image = 0; image < n; ++image) {
    int image_offset = image * width * height;
    float *d_out_offset = d_out + image_offset * n_filter; 
    float *d_in_offset  = d_in  + image_offset * depth;    

    for (int f = 0; f < n_filter; ++f) {
      for (int d = 0; d < depth; ++d) {
        float *filter_offset = filter + f * CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT * depth;

        for (int i = 0; i < height; ++i) {
          for (int j = 0; j < width; ++j) {
            float grad_val = d_out_offset[GET_1D_IDX(i, j, f, width, height)];

            for (int f_i = 0; f_i < CONV_FILTER_HEIGHT; ++f_i) {
              int row = i + f_i - CONV_FILTER_HEIGHT / 2;
              if (row < 0 || row >= height) continue; 

              for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {
                int col = j + f_j - CONV_FILTER_WIDTH / 2;
                if (col < 0 || col >= width) continue; 

                float w = filter_offset[GET_1D_IDX(f_i, f_j, d, CONV_FILTER_WIDTH, CONV_FILTER_HEIGHT)];
                d_in_offset[GET_1D_IDX(row, col, d, width, height)] += grad_val * w;
              }
            }
          }
        }
      }
    }
  }
}

// 14. UPDATE WEIGHT
void cpu_update_weight(float *weight, float *gradient, int size, float learning_rate) {
  for (int i = 0; i < size; ++i)
    weight[i] -= learning_rate * gradient[i];
}