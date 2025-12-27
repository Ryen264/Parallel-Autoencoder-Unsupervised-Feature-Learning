#include "optimized2_layers.h"

// =========================================================
// HELPER: Shared Memory Loading (Giữ nguyên)
// =========================================================
static __device__ void load_tile_to_shared(float *d_in,
                                           float *s_in,
                                           int    n_channels,
                                           int    width,
                                           int    height,
                                           int    i,
                                           int    j,
                                           int    tid_y,
                                           int    tid_x,
                                           int    dim_y,
                                           int    dim_x,
                                           int    padding_y,
                                           int    padding_x,
                                           int    shared_width,
                                           int    shared_height) {
  // 1. Center
  for (int c = 0; c < n_channels; ++c) {
    if (i < height && j < width) {
      s_in[GET_1D_IDX(
          tid_y + padding_y, tid_x + padding_x, c, shared_width, shared_height)] =
          d_in[GET_1D_IDX(i, j, c, width, height)];
    } else {
      s_in[GET_1D_IDX(
          tid_y + padding_y, tid_x + padding_x, c, shared_width, shared_height)] = 0.0f;
    }
  }

  // 2. Halo - Top & Bottom
  if (tid_y < padding_y) {
    int r_top = i - padding_y;
    if (r_top >= 0 && r_top < height && j < width) {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(tid_y, tid_x + padding_x, c, shared_width, shared_height)] =
            d_in[GET_1D_IDX(r_top, j, c, width, height)];
    } else {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(tid_y, tid_x + padding_x, c, shared_width, shared_height)] =
            0.0f;
    }

    int r_bot = i + dim_y;
    if (r_bot < height && j < width) {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(tid_y + dim_y + padding_y,
                        tid_x + padding_x,
                        c,
                        shared_width,
                        shared_height)] = d_in[GET_1D_IDX(r_bot, j, c, width, height)];
    } else {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(tid_y + dim_y + padding_y,
                        tid_x + padding_x,
                        c,
                        shared_width,
                        shared_height)] = 0.0f;
    }
  }

  // 3. Halo - Left & Right
  if (tid_x < padding_x) {
    int c_left = j - padding_x;
    if (c_left >= 0 && c_left < width && i < height) {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(tid_y + padding_y, tid_x, c, shared_width, shared_height)] =
            d_in[GET_1D_IDX(i, c_left, c, width, height)];
    } else {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(tid_y + padding_y, tid_x, c, shared_width, shared_height)] =
            0.0f;
    }

    int c_right = j + dim_x;
    if (c_right < width && i < height) {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(tid_y + padding_y,
                        tid_x + dim_x + padding_x,
                        c,
                        shared_width,
                        shared_height)] =
            d_in[GET_1D_IDX(i, c_right, c, width, height)];
    } else {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(tid_y + padding_y,
                        tid_x + dim_x + padding_x,
                        c,
                        shared_width,
                        shared_height)] = 0.0f;
    }
  }

  // 4. Corners
  if (tid_y < padding_y && tid_x < padding_x) {
    int r_tl = i - padding_y;
    int c_tl = j - padding_x;
    if (r_tl >= 0 && r_tl < height && c_tl >= 0 && c_tl < width) {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(tid_y, tid_x, c, shared_width, shared_height)] =
            d_in[GET_1D_IDX(r_tl, c_tl, c, width, height)];
    } else {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(tid_y, tid_x, c, shared_width, shared_height)] = 0.0f;
    }

    int r_tr = i - padding_y;
    int c_tr = j + dim_x;
    if (r_tr >= 0 && r_tr < height && c_tr < width) {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(
            tid_y, tid_x + dim_x + padding_x, c, shared_width, shared_height)] =
            d_in[GET_1D_IDX(r_tr, c_tr, c, width, height)];
    } else {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(
            tid_y, tid_x + dim_x + padding_x, c, shared_width, shared_height)] = 0.0f;
    }

    int r_bl = i + dim_y;
    int c_bl = j - padding_x;
    if (r_bl < height && c_bl >= 0 && c_bl < width) {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(
            tid_y + dim_y + padding_y, tid_x, c, shared_width, shared_height)] =
            d_in[GET_1D_IDX(r_bl, c_bl, c, width, height)];
    } else {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(
            tid_y + dim_y + padding_y, tid_x, c, shared_width, shared_height)] = 0.0f;
    }

    int r_br = i + dim_y;
    int c_br = j + dim_x;
    if (r_br < height && c_br < width) {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(tid_y + dim_y + padding_y,
                        tid_x + dim_x + padding_x,
                        c,
                        shared_width,
                        shared_height)] =
            d_in[GET_1D_IDX(r_br, c_br, c, width, height)];
    } else {
      for (int c = 0; c < n_channels; ++c)
        s_in[GET_1D_IDX(tid_y + dim_y + padding_y,
                        tid_x + dim_x + padding_x,
                        c,
                        shared_width,
                        shared_height)] = 0.0f;
    }
  }
}

// -------------------- Conv2D Forward --------------------
__global__ void optimized2_full_filter_kernel(float *in,
                                              float *filter,
                                              float *bias,
                                              float *in_relu,
                                              float *out,
                                              int    width,
                                              int    height,
                                              int    depth,
                                              int    n_filter) {
  extern __shared__ float s_in[];

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int tid_z = threadIdx.z;

  int j = blockIdx.x * blockDim.x + tid_x;
  int i = blockIdx.y * blockDim.y + tid_y;
  int f = blockIdx.z * blockDim.z + tid_z;

  int padding_x     = CONV_FILTER_WIDTH / 2;
  int padding_y     = CONV_FILTER_HEIGHT / 2;
  int shared_width  = blockDim.x + CONV_FILTER_WIDTH - 1;
  int shared_height = blockDim.y + CONV_FILTER_HEIGHT - 1;

  load_tile_to_shared(in,
                      s_in,
                      depth,
                      width,
                      height,
                      i,
                      j,
                      tid_y,
                      tid_x,
                      blockDim.y,
                      blockDim.x,
                      padding_y,
                      padding_x,
                      shared_width,
                      shared_height);

  __syncthreads();

  if (j >= width || i >= height || f >= n_filter)
    return;

  float  sum           = 0;
  float *filter_offset = filter + f * CONV_FILTER_HEIGHT * CONV_FILTER_WIDTH * depth;

  for (int f_i = 0; f_i < CONV_FILTER_HEIGHT; ++f_i) {
    for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {
      for (int d = 0; d < depth; ++d) {
        sum +=
            s_in[GET_1D_IDX(tid_y + f_i, tid_x + f_j, d, shared_width, shared_height)] *
            filter_offset[GET_1D_IDX(
                f_i, f_j, d, CONV_FILTER_WIDTH, CONV_FILTER_HEIGHT)];
      }
    }
  }

  int idx       = GET_1D_IDX(i, j, f, width, height);
  sum          += bias[f];
  in_relu[idx]  = sum;
  out[idx]      = fmaxf(0.0f, sum);
}

// -------------------- Max Pooling (2x down) --------------------
__global__ void
optimized2_max_pooling_kernel(float *in, float *out, int width, int height, int depth) {
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
      fmaxf(fmaxf(in[GET_1D_IDX(in_y, in_x, d, width, height)],
                  in[GET_1D_IDX(in_y, in_x + 1, d, width, height)]),
            fmaxf(in[GET_1D_IDX(in_y + 1, in_x, d, width, height)],
                  in[GET_1D_IDX(in_y + 1, in_x + 1, d, width, height)]));
}

// -------------------- Upsampling (2x up) --------------------
__global__ void
optimized2_upsampling_kernel(float *in, float *out, int width, int height, int depth) {
  int i = blockIdx.y * blockDim.y + threadIdx.y; // Output Height
  int j = blockIdx.x * blockDim.x + threadIdx.x; // Output Width
  int d = blockIdx.z * blockDim.z + threadIdx.z; // Depth

  int new_width  = width * 2;
  int new_height = height * 2;

  if (j >= new_width || i >= new_height || d >= depth)
    return;

  int in_i = i / 2;
  int in_j = j / 2;
  out[GET_1D_IDX(i, j, d, new_width, new_height)] =
      in[GET_1D_IDX(in_i, in_j, d, width, height)];
}

// -------------------- Upsampling Backward --------------------
__global__ void optimized2_upsampling_backward_kernel(
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
optimized2_mse_loss_kernel(float *expected, float *actual, float *out, int size) {
  __shared__ float shared[MAX_BLOCK_SIZE];
  int              tid    = threadIdx.x;
  int              offset = (blockDim.x * blockIdx.x) * 2 + tid;
  shared[tid]             = 0;

  if (offset < size) {
    shared[tid] = SQR(expected[offset] - actual[offset]);
    if (blockDim.x > 32 && (offset += blockDim.x) < size)
      shared[tid] += SQR(expected[offset] - actual[offset]);
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s)
      shared[tid] += shared[tid + s];
    __syncthreads();
  }

  if (tid < 32) {
    volatile float *vmem = shared;
    if (blockDim.x >= 64)
      vmem[tid] = vmem[tid] + vmem[tid + 32];
    vmem[tid] = vmem[tid] + vmem[tid + 16];
    vmem[tid] = vmem[tid] + vmem[tid + 8];
    vmem[tid] = vmem[tid] + vmem[tid + 4];
    vmem[tid] = vmem[tid + 2];
    vmem[tid] = vmem[tid + 1];
  }

  if (tid == 0)
    atomicAdd(out, shared[0] / size);
}

__global__ void
optimized2_mse_grad_kernel(float *expected, float *actual, float *d_out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    d_out[idx] = 2.0f * (actual[idx] - expected[idx]) / size;
}

// -------------------- ReLU Backward --------------------
__global__ void
optimized2_relu_backward_kernel(float *in, float *d_out, float *d_in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    d_in[idx] = in[idx] > 0 ? d_out[idx] : 0;
}

// -------------------- Weight Update --------------------
__global__ void optimized2_update_weight_kernel(float *weight,
                                                float *gradient,
                                                int    size,
                                                float  learning_rate) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    weight[idx] -= learning_rate * gradient[idx];
}

// -------------------- Conv2D Backward: Filter Gradient only --------------------
__global__ void optimized2_full_filter_grad_kernel(float *in,
                                                   float *d_out,
                                                   float *d_bias,
                                                   float *d_filter,
                                                   int    width,
                                                   int    height,
                                                   int    depth,
                                                   int    n_filter) {
  extern __shared__ float s_in[];

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int tid_z = threadIdx.z;

  int j = blockIdx.x * blockDim.x + tid_x;
  int i = blockIdx.y * blockDim.y + tid_y;
  int f = blockIdx.z * blockDim.z + tid_z; // Filter Index

  int padding_x     = CONV_FILTER_WIDTH / 2;
  int padding_y     = CONV_FILTER_HEIGHT / 2;
  int shared_width  = blockDim.x + CONV_FILTER_WIDTH - 1;
  int shared_height = blockDim.y + CONV_FILTER_HEIGHT - 1;

  load_tile_to_shared(in,
                      s_in,
                      depth,
                      width,
                      height,
                      i,
                      j,
                      tid_y,
                      tid_x,
                      blockDim.y,
                      blockDim.x,
                      padding_y,
                      padding_x,
                      shared_width,
                      shared_height);
  __syncthreads();

  if (j >= width || i >= height || f >= n_filter)
    return;

  float *d_filter_offset =
      d_filter + f * CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT * depth;
  float d_out_val = d_out[GET_1D_IDX(i, j, f, width, height)];
  atomicAdd(d_bias + f, d_out_val);

  for (int f_i = 0; f_i < CONV_FILTER_HEIGHT; ++f_i) {
    for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {
      for (int d = 0; d < depth; ++d) {
        float val_in =
            s_in[GET_1D_IDX(tid_y + f_i, tid_x + f_j, d, shared_width, shared_height)];
        atomicAdd(d_filter_offset +
                      GET_1D_IDX(f_i, f_j, d, CONV_FILTER_WIDTH, CONV_FILTER_HEIGHT),
                  val_in * d_out_val);
      }
    }
  }
}

__global__ void optimized2_conv2D_backward_kernel(float *d_out,
                                                  float *filter,
                                                  float *d_in,
                                                  int    width,
                                                  int    height,
                                                  int    depth,
                                                  int    n_filter) {
  extern __shared__ float s_in[]; // Chứa tile của 'd_out'

  int tid_x = threadIdx.x;
  int tid_y = threadIdx.y;
  int tid_z = threadIdx.z;

  int j = blockIdx.x * blockDim.x + tid_x;
  int i = blockIdx.y * blockDim.y + tid_y;
  int d = blockIdx.z * blockDim.z + tid_z;

  int padding_x     = CONV_FILTER_WIDTH / 2;
  int padding_y     = CONV_FILTER_HEIGHT / 2;
  int shared_width  = blockDim.x + CONV_FILTER_WIDTH - 1;
  int shared_height = blockDim.y + CONV_FILTER_HEIGHT - 1;

  load_tile_to_shared(d_out,
                      s_in,
                      n_filter,
                      width,
                      height,
                      i,
                      j,
                      tid_y,
                      tid_x,
                      blockDim.y,
                      blockDim.x,
                      padding_y,
                      padding_x,
                      shared_width,
                      shared_height);
  __syncthreads();

  if (j >= width || i >= height || d >= depth)
    return;

  float d_sum = 0;

  for (int f = 0; f < n_filter; ++f) {
    float *filter_offset = filter + f * CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT * depth;

    for (int f_i = 0; f_i < CONV_FILTER_HEIGHT; ++f_i) {
      for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {

        // *** KEY FIX: FLIP INDICES FOR CONVOLUTION BACKPROP ***
        // Use (CONV_FILTER_HEIGHT - 1 - f_i) instead of f_i to perform proper
        // convolution

        float val_dout =
            s_in[GET_1D_IDX(tid_y + f_i, tid_x + f_j, f, shared_width, shared_height)];

        float val_filter = filter_offset[GET_1D_IDX(CONV_FILTER_HEIGHT - 1 - f_i,
                                                    CONV_FILTER_WIDTH - 1 - f_j,
                                                    d, // depth
                                                    CONV_FILTER_WIDTH,
                                                    CONV_FILTER_HEIGHT)];

        d_sum += val_filter * val_dout;
      }
    }
  }

  d_in[GET_1D_IDX(i, j, d, width, height)] = d_sum;
}

// optimized2 Max Pooling Backward
__global__ void optimized2_max_pooling_backward_kernel(
    float *in, float *d_out, float *d_in, int width, int height, int depth) {
  int i = blockIdx.y * blockDim.y + threadIdx.y; // y
  int j = blockIdx.x * blockDim.x + threadIdx.x; // x
  int d = blockIdx.z * blockDim.z + threadIdx.z;

  if (j >= width || i >= height || d >= depth)
    return;

  int idx        = GET_1D_IDX(i, j, d, width, height);
  int out_i      = i / 2;
  int out_j      = j / 2;
  int new_width  = width / 2;
  int new_height = height / 2;

  // Indices of the 2x2 block contributing to pooled output (out_i, out_j)
  int base_y          = out_i * 2;
  int base_x          = out_j * 2;
  int neighbors_idx[] = {
    GET_1D_IDX(base_y, base_x, d, width, height),
    GET_1D_IDX(base_y, base_x + 1, d, width, height),
    GET_1D_IDX(base_y + 1, base_x, d, width, height),
    GET_1D_IDX(base_y + 1, base_x + 1, d, width, height),
  };

  // Find max index manually (device-safe)
  int   max_idx = neighbors_idx[0];
  float max_val = in[max_idx];
  for (int k = 1; k < 4; ++k) {
    int   n_idx = neighbors_idx[k];
    float n_val = in[n_idx];
    if (n_val > max_val) {
      max_val = n_val;
      max_idx = n_idx;
    }
  }

  d_in[idx] = (idx == max_idx)
                  ? d_out[GET_1D_IDX(out_i, out_j, d, new_width, new_height)]
                  : 0.0f;
}

void optimized2_full_filter(float *in,
                            float *filter,
                            float *bias,
                            float *in_relu,
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
                    depth *
                    sizeof(float);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * n_filter;

    optimized2_full_filter_kernel<<<grid_size, block_size, shared_size>>>(
        in + in_offset,
        filter,
        bias,
        in_relu + out_offset,
        out + out_offset,
        width,
        height,
        depth,
        n_filter);
  }
}

void optimized2_max_pooling(
    float *in, float *out, int n, int width, int height, int depth, dim3 block_size) {
  dim3 grid_size((width / 2 - 1) / block_size.x + 1,
                 (height / 2 - 1) / block_size.y + 1,
                 (depth - 1) / block_size.z + 1);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * depth / 4;

    optimized2_max_pooling_kernel<<<grid_size, block_size>>>(
        in + in_offset, out + out_offset, width, height, depth);
  }
}

void optimized2_upsampling(
    float *in, float *out, int n, int width, int height, int depth, dim3 block_size) {
  dim3 grid_size((2 * width - 1) / block_size.x + 1,
                 (2 * height - 1) / block_size.y + 1,
                 (depth - 1) / block_size.z + 1);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * depth * 4;

    optimized2_upsampling_kernel<<<grid_size, block_size>>>(
        in + in_offset, out + out_offset, width, height, depth);
  }
}

float optimized2_mse_loss(float *expected,
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

  optimized2_mse_loss_kernel<<<grid_size, block_size>>>(expected, actual, d_loss, size);

  CUDA_CHECK(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_loss));

  return loss;
}

void optimized2_mse_grad(float *expected,
                         float *actual,
                         float *d_out,
                         int    n,
                         int    width,
                         int    height,
                         int    depth,
                         dim3   block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((size - 1) / block_size.x + 1);

  optimized2_mse_grad_kernel<<<grid_size, block_size>>>(expected, actual, d_out, size);
}

void optimized2_relu_backward(float *in,
                              float *d_out,
                              float *d_in,
                              int    n,
                              int    width,
                              int    height,
                              int    depth,
                              dim3   block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((size - 1) / block_size.x + 1);

  optimized2_relu_backward_kernel<<<grid_size, block_size>>>(in, d_out, d_in, size);
}

void optimized2_max_pooling_backward(float *in,
                                     float *d_out,
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

    optimized2_max_pooling_backward_kernel<<<grid_size, block_size>>>(
        in + in_offset, d_out + out_offset, d_in + in_offset, width, height, depth);
  }
}

void optimized2_upsampling_backward(float *d_out,
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

    optimized2_upsampling_backward_kernel<<<grid_size, block_size>>>(
        d_out + out_offset, d_in + in_offset, width, height, depth);
  }
}

void optimized2_full_filter_grad(float *in,
                                 float *d_out,
                                 float *d_bias,
                                 float *d_filter,
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
                    depth *
                    sizeof(float);

  CUDA_CHECK(cudaMemset(d_bias, 0, n_filter * sizeof(float)));
  CUDA_CHECK(cudaMemset(
      d_filter,
      0,
      CONV_FILTER_HEIGHT * CONV_FILTER_WIDTH * depth * n_filter * sizeof(float)));

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * n_filter;

    optimized2_full_filter_grad_kernel<<<grid_size, block_size, shared_size>>>(
        in + in_offset,
        d_out + out_offset,
        d_bias,
        d_filter,
        width,
        height,
        depth,
        n_filter);
  }
}

void optimized2_update_weight(
    float *weight, float *gradient, int size, float learning_rate, dim3 block_size) {
  dim3 grid_size((size - 1) / block_size.x + 1);

  optimized2_update_weight_kernel<<<grid_size, block_size>>>(
      weight, gradient, size, learning_rate);
}

void optimized2_conv2D_backward(float *d_out,
                                float *filter,
                                float *d_in,
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
                    n_filter *
                    sizeof(float);

  for (int i = 0; i < n; ++i) {
    int d_in_offset  = i * width * height * depth;
    int d_out_offset = i * width * height * n_filter;

    optimized2_conv2D_backward_kernel<<<grid_size, block_size, shared_size>>>(
        d_out + d_out_offset,
        filter,
        d_in + d_in_offset,
        width,
        height,
        depth,
        n_filter);
  }
}