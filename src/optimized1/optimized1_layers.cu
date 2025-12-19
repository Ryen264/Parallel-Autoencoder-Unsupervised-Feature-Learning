#include "optimized1_layers.h"

// -------------------- Conv2D Forward --------------------
__global__ void optimized1_conv2D_kernel(float *in,
                                         float *filter,
                                         float *out,
                                         int    width,
                                         int    height,
                                         int    depth,
                                         int    n_filter) {
  extern __shared__ float s_in[];

  int tid_y = threadIdx.y;
  int tid_x = threadIdx.x;
  int tid_z = threadIdx.z;
  int dim_y = blockDim.y;
  int dim_x = blockDim.x;
  int i     = blockIdx.y * dim_y + tid_y;
  int j     = blockIdx.x * dim_x + tid_x;
  int f     = tid_z * blockDim.z + threadIdx.z;

  if (j >= width || i >= height || f >= n_filter)
    return;

  int    padding_y     = CONV_FILTER_HEIGHT / 2;
  int    padding_x     = CONV_FILTER_WIDTH / 2;
  int    shared_y      = tid_y + padding_y;
  int    shared_x      = tid_x + padding_x;
  int    shared_height = dim_y + CONV_FILTER_HEIGHT - 1;
  int    shared_width  = dim_x + CONV_FILTER_WIDTH - 1;
  float *filter_offset = filter + f * CONV_FILTER_HEIGHT * CONV_FILTER_WIDTH * depth;
  float  sum           = 0;

  for (int d = 0; d < depth; ++d) {
    if (tid_x == 0 && tid_y == 0 && tid_z == 0)
      memset(s_in, 0, shared_height * shared_width);
    __syncthreads();

    s_in[GET_1D_IDX_2D(shared_y, shared_x, shared_width)] =
        in[GET_1D_IDX(i, j, d, width, height)];

    if (tid_y == 0) {
      for (int f_i = 0; f_i < padding_y; ++f_i) {
        int cur_row = i - padding_y + f_i;
        if (cur_row >= 0)
          s_in[GET_1D_IDX_2D(f_i, shared_x, shared_width)] =
              in[GET_1D_IDX(cur_row, j, d, width, height)];

        if (tid_x == 0) {
          for (int f_j = 0; f_j < padding_x; ++f_j) {
            int cur_col = j - padding_x + f_j;
            if (cur_col >= 0)
              s_in[GET_1D_IDX_2D(f_i, f_j, shared_width)] =
                  in[GET_1D_IDX(cur_row, cur_col, d, width, height)];
          }
        }

        if (tid_x + 1 == dim_x) {
          for (int f_j = 1; f_j <= padding_x; ++f_j) {
            int cur_col = j + f_j;
            if (cur_col < width)
              s_in[GET_1D_IDX_2D(f_i, shared_x + f_j, shared_width)] =
                  in[GET_1D_IDX(cur_row, cur_col, d, width, height)];
          }
        }
      }
    }

    if (tid_y + 1 == dim_y) {
      for (int f_i = 1; f_i <= padding_y; ++f_i) {
        int cur_row = i + f_i;
        if (cur_row < height)
          s_in[GET_1D_IDX_2D(shared_y + f_i, shared_x, shared_width)] =
              in[GET_1D_IDX(cur_row, j, d, width, height)];

        if (tid_x == 0) {
          for (int f_j = 0; f_j < padding_x; ++f_j) {
            int cur_col = j - padding_x + f_j;
            if (cur_col >= 0)
              s_in[GET_1D_IDX_2D(shared_y + f_i, f_j, shared_width)] =
                  in[GET_1D_IDX(cur_row, cur_col, d, width, height)];
          }
        }

        if (tid_x + 1 == dim_x) {
          for (int f_j = 1; f_j <= padding_x; ++f_j) {
            int cur_col = j + f_j;
            if (cur_col < width)
              s_in[GET_1D_IDX_2D(shared_y + f_i, shared_x + f_j, shared_width)] =
                  in[GET_1D_IDX(cur_row, cur_col, d, width, height)];
          }
        }
      }
    }

    if (tid_x == 0) {
      for (int f_j = 0; f_j < padding_x; ++f_j) {
        int cur_col = j - padding_x + f_j;
        if (cur_col >= 0)
          s_in[GET_1D_IDX_2D(shared_y, f_j, shared_width)] =
              in[GET_1D_IDX(i, cur_col, d, width, height)];
      }
    }

    if (tid_x + 1 == dim_x) {
      for (int f_j = 1; f_j <= padding_x; ++f_j) {
        int cur_col = j + f_j;
        if (cur_col < width)
          s_in[GET_1D_IDX_2D(shared_y, shared_x + f_j, shared_width)] =
              in[GET_1D_IDX(i, cur_col, d, width, height)];
      }
    }

    __syncthreads();

    for (int f_i = 0; f_i < CONV_FILTER_HEIGHT; ++f_i) {
      for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {
        sum += s_in[GET_1D_IDX_2D(tid_y + f_i, tid_x + f_j, shared_width)] *
               filter_offset[GET_1D_IDX(
                   f_i, f_j, d, CONV_FILTER_WIDTH, CONV_FILTER_HEIGHT)];
      }
    }
  }

  out[GET_1D_IDX(i, j, f, width, height)] = sum;
}

// -------------------- Bias --------------------
__global__ void optimized1_add_bias_kernel(
    float *in, float *bias, float *out, int n, int img_size, int depth) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * img_size * depth)
    out[idx] = in[idx] + bias[(idx % (img_size * depth)) / img_size];
}

// -------------------- ReLU --------------------
__global__ void optimized1_relu_kernel(float *in, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    out[idx] = fmaxf(0.0f, in[idx]);
}

// -------------------- Max Pooling (2x down) --------------------
__global__ void
optimized1_avg_pooling_kernel(float *in, float *out, int width, int height, int depth) {
  int x          = blockIdx.x * blockDim.x + threadIdx.x;
  int y          = blockIdx.y * blockDim.y + threadIdx.y;
  int d          = blockIdx.z * blockDim.z + threadIdx.z;
  int new_width  = width / 2;
  int new_height = height / 2;

  if (x >= new_width || y >= new_height || d >= depth)
    return;

  int in_x = x * 2;
  int in_y = y * 2;

  out[GET_1D_IDX(y, x, d, new_width, new_height)] =
      (in[GET_1D_IDX(in_y, in_x, d, width, height)] +
       in[GET_1D_IDX(in_y, in_x + 1, d, width, height)] +
       in[GET_1D_IDX(in_y + 1, in_x, d, width, height)] +
       in[GET_1D_IDX(in_y + 1, in_x + 1, d, width, height)]) /
      4.0f;
}

// -------------------- Upsampling (2x up) --------------------
__global__ void
optimized1_upsampling_kernel(float *in, float *out, int width, int height, int depth) {
  int x          = blockIdx.x * blockDim.x + threadIdx.x;
  int y          = blockIdx.y * blockDim.y + threadIdx.y;
  int d          = blockIdx.z * blockDim.z + threadIdx.z;
  int new_width  = width * 2;
  int new_height = height * 2;

  if (x >= width || y >= height || d >= depth)
    return;

  int   out_x = 2 * x;
  int   out_y = 2 * y;
  float val   = in[GET_1D_IDX(y, x, d, width, height)];

  out[GET_1D_IDX(out_y, out_x, d, new_width, new_height)]         = val;
  out[GET_1D_IDX(out_y, out_x + 1, d, new_width, new_height)]     = val;
  out[GET_1D_IDX(out_y + 1, out_x, d, new_width, new_height)]     = val;
  out[GET_1D_IDX(out_y + 1, out_x + 1, d, new_width, new_height)] = val;
}

// -------------------- Upsampling Backward --------------------
__global__ void optimized1_upsampling_backward_kernel(
    float *d_out, float *d_in, int width, int height, int depth) {
  int x          = blockIdx.x * blockDim.x + threadIdx.x;
  int y          = blockIdx.y * blockDim.y + threadIdx.y;
  int d          = blockIdx.z * blockDim.z + threadIdx.z;
  int new_width  = width * 2;
  int new_height = height * 2;

  if (x >= width || y >= height || d >= depth)
    return;

  int   out_x = 2 * x;
  int   out_y = 2 * y;
  float d_sum = 0;

  d_sum += d_out[GET_1D_IDX(out_y, out_x, d, new_width, new_height)];
  d_sum += d_out[GET_1D_IDX(out_y, out_x + 1, d, new_width, new_height)];
  d_sum += d_out[GET_1D_IDX(out_y + 1, out_x, d, new_width, new_height)];
  d_sum += d_out[GET_1D_IDX(out_y + 1, out_x + 1, d, new_width, new_height)];

  d_in[GET_1D_IDX(y, x, d, width, height)] = d_sum;
}

// -------------------- MSE Loss & Gradient --------------------
__global__ void
optimized1_mse_loss_kernel(float *expected, float *actual, float *out, int size) {
  __shared__ float shared[MAX_BLOCK_SIZE];

  int    tid    = threadIdx.x;
  float *elem   = shared + tid;
  int    offset = (blockDim.x * blockIdx.x) * 2 + tid;

  // Set all the element to 0
  *elem = 0;

  // Copy into shared memory, and doing the first pass at the same time
  if (offset < size) {
    *elem = SQR(expected[offset] - actual[offset]);
    // If block size is less than 32,
    // we do the unroll directly
    if (blockDim.x > 32 && (offset += blockDim.x) < size)
      *elem += SQR(expected[offset] - actual[offset]);
  }
  __syncthreads();

  // Since MAX_BLOCK_SIZE is 1024, we start from 512
  // because we already did the first pass
  if (blockDim.x > 512) {
    if (tid < 512)
      *elem += elem[512];
    __syncthreads();
  }

  if (blockDim.x > 256) {
    if (tid < 256)
      *elem += elem[256];
    __syncthreads();
  }

  if (blockDim.x > 128) {
    if (tid < 128)
      *elem += elem[128];
    __syncthreads();
  }

  if (blockDim.x > 64) {
    if (tid < 64)
      *elem += elem[64];
    __syncthreads();
  }

  // Unroll last warp
  if (tid < 32) {
    volatile float *vmem = elem;

    *vmem += vmem[32];
    *vmem += vmem[16];
    *vmem += vmem[8];
    *vmem += vmem[4];
    *vmem += vmem[2];
    *vmem += vmem[1];
  }

  // Copy to output
  if (tid == 0)
    atomicAdd(out, *elem / size);
}

__global__ void
optimized1_mse_grad_kernel(float *expected, float *actual, float *d_out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    d_out[idx] = 2.0f * (actual[idx] - expected[idx]) / size;
}

// -------------------- ReLU Backward --------------------
__global__ void
optimized1_relu_backward_kernel(float *in, float *d_out, float *d_in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    d_out[idx] = in[idx] > 0 ? d_in[idx] : 0;
}

// -------------------- Bias Gradient --------------------
__global__ void optimized1_bias_grad_kernel(
    float *d_out, float *d_bias, int n, int img_size, int depth) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n * img_size * depth)
    atomicAdd(d_bias + (idx % (img_size * depth) / img_size), d_out[idx]);
}

// -------------------- Weight Update --------------------
__global__ void optimized1_update_weight_kernel(float *weight,
                                                float *gradient,
                                                int    size,
                                                float  learning_rate) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    weight[idx] -= learning_rate * gradient[idx];
}

// -------------------- Conv2D Backward: Filter Gradient only --------------------
__global__ void optimized1_conv2D_grad_kernel(float *in,
                                              float *d_out,
                                              float *d_filter,
                                              int    width,
                                              int    height,
                                              int    depth,
                                              int    n_filter) {
  extern __shared__ float s_in[];

  int tid_y = threadIdx.y;
  int tid_x = threadIdx.x;
  int tid_z = threadIdx.z;
  int dim_y = blockDim.y;
  int dim_x = blockDim.x;
  int i     = blockIdx.y * dim_y + tid_y;
  int j     = blockIdx.x * dim_x + tid_x;
  int f     = tid_z * blockDim.z + threadIdx.z;

  if (j >= width || i >= height || f >= n_filter)
    return;

  int    padding_y     = CONV_FILTER_HEIGHT / 2;
  int    padding_x     = CONV_FILTER_WIDTH / 2;
  int    shared_y      = tid_y + padding_y;
  int    shared_x      = tid_x + padding_x;
  int    shared_height = dim_y + CONV_FILTER_HEIGHT - 1;
  int    shared_width  = dim_x + CONV_FILTER_WIDTH - 1;
  float  sum           = 0;
  float *d_filter_offset =
      d_filter + f * CONV_FILTER_HEIGHT * CONV_FILTER_WIDTH * depth;
  float d_out_val = d_out[GET_1D_IDX(i, j, f, width, height)];

  for (int d = 0; d < depth; ++d) {
    if (tid_x == 0 && tid_y == 0 && tid_z == 0)
      memset(s_in, 0, shared_height * shared_width);
    __syncthreads();

    s_in[GET_1D_IDX_2D(shared_y, shared_x, shared_width)] =
        in[GET_1D_IDX(i, j, d, width, height)];

    if (tid_y == 0) {
      for (int f_i = 0; f_i < padding_y; ++f_i) {
        int cur_row = i - padding_y + f_i;
        if (cur_row >= 0)
          s_in[GET_1D_IDX_2D(f_i, shared_x, shared_width)] =
              in[GET_1D_IDX(cur_row, j, d, width, height)];

        if (tid_x == 0) {
          for (int f_j = 0; f_j < padding_x; ++f_j) {
            int cur_col = j - padding_x + f_j;
            if (cur_col >= 0)
              s_in[GET_1D_IDX_2D(f_i, f_j, shared_width)] =
                  in[GET_1D_IDX(cur_row, cur_col, d, width, height)];
          }
        }

        if (tid_x + 1 == dim_x) {
          for (int f_j = 1; f_j <= padding_x; ++f_j) {
            int cur_col = j + f_j;
            if (cur_col < width)
              s_in[GET_1D_IDX_2D(f_i, shared_x + f_j, shared_width)] =
                  in[GET_1D_IDX(cur_row, cur_col, d, width, height)];
          }
        }
      }
    }

    if (tid_y + 1 == dim_y) {
      for (int f_i = 1; f_i <= padding_y; ++f_i) {
        int cur_row = i + f_i;
        if (cur_row < height)
          s_in[GET_1D_IDX_2D(shared_y + f_i, shared_x, shared_width)] =
              in[GET_1D_IDX(cur_row, j, d, width, height)];

        if (tid_x == 0) {
          for (int f_j = 0; f_j < padding_x; ++f_j) {
            int cur_col = j - padding_x + f_j;
            if (cur_col >= 0)
              s_in[GET_1D_IDX_2D(shared_y + f_i, f_j, shared_width)] =
                  in[GET_1D_IDX(cur_row, cur_col, d, width, height)];
          }
        }

        if (tid_x + 1 == dim_x) {
          for (int f_j = 1; f_j <= padding_x; ++f_j) {
            int cur_col = j + f_j;
            if (cur_col < width)
              s_in[GET_1D_IDX_2D(shared_y + f_i, shared_x + f_j, shared_width)] =
                  in[GET_1D_IDX(cur_row, cur_col, d, width, height)];
          }
        }
      }
    }

    if (tid_x == 0) {
      for (int f_j = 0; f_j < padding_x; ++f_j) {
        int cur_col = j - padding_x + f_j;
        if (cur_col >= 0)
          s_in[GET_1D_IDX_2D(shared_y, f_j, shared_width)] =
              in[GET_1D_IDX(i, cur_col, d, width, height)];
      }
    }

    if (tid_x + 1 == dim_x) {
      for (int f_j = 1; f_j <= padding_x; ++f_j) {
        int cur_col = j + f_j;
        if (cur_col < width)
          s_in[GET_1D_IDX_2D(shared_y, shared_x + f_j, shared_width)] =
              in[GET_1D_IDX(i, cur_col, d, width, height)];
      }
    }

    __syncthreads();

    for (int f_i = 0; f_i < CONV_FILTER_HEIGHT; ++f_i) {
      for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {
        atomicAdd(d_filter_offset +
                      GET_1D_IDX(f_i, f_j, d, CONV_FILTER_WIDTH, CONV_FILTER_HEIGHT),
                  s_in[GET_1D_IDX_2D(tid_y + f_i, tid_x + f_j, shared_width)] *
                      d_out_val);
      }
    }
  }
}

// optimized1 Max Pooling Backward
__global__ void optimized1_avg_pooling_backward_kernel(
    float *d_out, float *d_in, int width, int height, int depth) {
  int x     = blockIdx.x * blockDim.x + threadIdx.x;
  int y     = blockIdx.y * blockDim.y + threadIdx.y;
  int d     = blockIdx.z * blockDim.z + threadIdx.z;
  int out_x = x / 2;
  int out_y = y / 2;

  if (x >= width || y >= height || d >= depth)
    return;

  d_in[GET_1D_IDX(y, x, d, width, height)] =
      d_out[GET_1D_IDX(out_y, out_x, d, width / 2, height / 2)] / 4.0f;
}

void optimized1_conv2D(float *in,
                       float *filter,
                       float *out,
                       int    n,
                       int    width,
                       int    height,
                       int    depth,
                       int    n_filter,
                       dim3   block_size) {
  dim3 grid_size((width - 1) / block_size.x + 1,
                 (height - 1) / block_size.y + 1,
                 (n_filter - 1) / block_size.z + 1);
  int  shared_size = (block_size.x + CONV_FILTER_WIDTH - 1) *
                    (block_size.y + CONV_FILTER_HEIGHT - 1) *
                    sizeof(float);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * n_filter;

    optimized1_conv2D_kernel<<<grid_size, block_size, shared_size>>>(
        in + in_offset, filter, out + out_offset, width, height, depth, n_filter);
  }
}

void optimized1_add_bias(float *in,
                         float *bias,
                         float *out,
                         int    n,
                         int    width,
                         int    height,
                         int    depth,
                         dim3   block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((size - 1) / block_size.x + 1);

  optimized1_add_bias_kernel<<<grid_size, block_size>>>(
      in, bias, out, n, width * height, depth);
}

void optimized1_relu(
    float *in, float *out, int n, int width, int height, int depth, dim3 block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((size - 1) / block_size.x + 1);

  optimized1_relu_kernel<<<grid_size, block_size>>>(in, out, size);
}

void optimized1_avg_pooling(
    float *in, float *out, int n, int width, int height, int depth, dim3 block_size) {
  dim3 grid_size((width / 2 - 1) / block_size.x + 1,
                 (height / 2 - 1) / block_size.y + 1,
                 (depth - 1) / block_size.z + 1);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * depth / 4;

    optimized1_avg_pooling_kernel<<<grid_size, block_size>>>(
        in + in_offset, out + out_offset, width, height, depth);
  }
}

void optimized1_upsampling(
    float *in, float *out, int n, int width, int height, int depth, dim3 block_size) {
  dim3 grid_size((width - 1) / block_size.x + 1,
                 (height - 1) / block_size.y + 1,
                 (depth - 1) / block_size.z + 1);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * depth * 4;

    optimized1_upsampling_kernel<<<grid_size, block_size>>>(
        in + in_offset, out + out_offset, width, height, depth);
  }
}

float optimized1_mse_loss(float *expected,
                          float *actual,
                          int    n,
                          int    width,
                          int    height,
                          int    depth,
                          dim3   block_size) {
  int    size = n * width * height * depth;
  dim3   grid_size((size - 1) / block_size.x + 1);
  float  loss = 0;
  float *d_loss;
  CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

  optimized1_mse_loss_kernel<<<grid_size, block_size>>>(expected, actual, d_loss, size);
  CUDA_CHECK(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_loss));

  return loss;
}

void optimized1_mse_grad(float *expected,
                         float *actual,
                         float *d_out,
                         int    n,
                         int    width,
                         int    height,
                         int    depth,
                         dim3   block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((size - 1) / block_size.x + 1);

  optimized1_mse_grad_kernel<<<grid_size, block_size>>>(expected, actual, d_out, size);
}

void optimized1_relu_backward(float *in,
                              float *d_out,
                              float *d_in,
                              int    n,
                              int    width,
                              int    height,
                              int    depth,
                              dim3   block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((block_size.x + size - 1) / block_size.x + 1);

  optimized1_relu_backward_kernel<<<grid_size, block_size>>>(in, d_out, d_in, size);
}

void optimized1_avg_pooling_backward(float *d_out,
                                     float *d_in,
                                     int    n,
                                     int    width,
                                     int    height,
                                     int    depth,
                                     dim3   block_size) {
  dim3 grid_size((width - 1) / block_size.x + 1,
                 (height - 1) / block_size.y + 1,
                 (depth - 1) / block_size.z + 1);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * depth / 4;

    optimized1_avg_pooling_backward_kernel<<<grid_size, block_size>>>(
        d_out + out_offset, d_in + in_offset, width, height, depth);
  }
}

void optimized1_upsampling_backward(float *d_out,
                                    float *d_in,
                                    int    n,
                                    int    width,
                                    int    height,
                                    int    depth,
                                    dim3   block_size) {
  dim3 grid_size((width - 1) / block_size.x + 1,
                 (height - 1) / block_size.y + 1,
                 (depth - 1) / block_size.z + 1);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * depth * 4;

    optimized1_upsampling_backward_kernel<<<grid_size, block_size>>>(
        d_out + out_offset, d_in + in_offset, width, height, depth);
  }
}

void optimized1_bias_grad(float *d_out,
                          float *d_bias,
                          int    n,
                          int    width,
                          int    height,
                          int    depth,
                          dim3   block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((size - 1) / block_size.x + 1);

  CUDA_CHECK(cudaMemset(d_bias, 0, depth * sizeof(float)));
  optimized1_bias_grad_kernel<<<grid_size, block_size>>>(
      d_out, d_bias, n, width * height, depth);
}

void optimized1_conv2D_grad(float *in,
                            float *d_out,
                            float *d_filter,
                            int    n,
                            int    width,
                            int    height,
                            int    depth,
                            int    n_filter,
                            dim3   block_size) {
  dim3 grid_size((width - 1) / block_size.x + 1,
                 (height - 1) / block_size.y + 1,
                 (depth - 1) / block_size.z + 1);
  int  shared_size = (block_size.x + CONV_FILTER_WIDTH - 1) *
                    (block_size.y + CONV_FILTER_HEIGHT - 1) *
                    sizeof(float);

  CUDA_CHECK(cudaMemset(
      d_filter,
      0,
      CONV_FILTER_HEIGHT * CONV_FILTER_WIDTH * depth * n_filter * sizeof(float)));

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * n_filter;

    optimized1_conv2D_grad_kernel<<<grid_size, block_size, shared_size>>>(
        in + in_offset, d_out + out_offset, d_filter, width, height, depth, n_filter);
  }
}

void optimized1_update_weight(
    float *weight, float *gradient, int size, float learning_rate, dim3 block_size) {
  dim3 grid_size((size - 1) / block_size.x + 1);
  optimized1_update_weight_kernel<<<grid_size, block_size>>>(
      weight, gradient, size, learning_rate);
}