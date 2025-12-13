#include "constants.h"
#include "gpu_layers.h"
#include "macro.h"
#include <algorithm>

// -------------------- Conv2D Forward --------------------
__global__ void gpu_conv2D_kernel(float *in,
                                  float *filter,
                                  float *out,
                                  int    width,
                                  int    height,
                                  int    depth,
                                  int    n_filter) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int f = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= width || y >= height || f >= n_filter)
    return;

  int padding_x = CONV_FILTER_WIDTH / 2;
  int padding_y = CONV_FILTER_HEIGHT / 2;

  float sum = 0;
  for (int f_i = 0; f_i < CONV_FILTER_HEIGHT; ++f_i) {
    // If the row needs padding, we skip since we pad with 0
    int row = y + f_i - CONV_FILTER_HEIGHT / 2;
    if (row < 0 || row >= height)
      continue;

    for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {
      // Same with column
      int col = x + f_j - CONV_FILTER_WIDTH / 2;
      if (col < 0 || col >= width)
        continue;

      // Calculate start of filter
      float *cur_filter =
          filter + f * GET_1D_IDX(f_i, f_j, 0, CONV_FILTER_WIDTH, depth);

      // Calculate start of input
      float *in_start = in + GET_1D_IDX(row, col, 0, width, depth);

      for (int d = 0; d < depth; ++d)
        sum += in_start[d] * cur_filter[d];
    }
  }

  out[GET_1D_IDX(y, x, d, width, depth)] = sum;
}

// -------------------- Bias --------------------
__global__ void gpu_add_bias_kernel(float *in, float *bias, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    out[idx] = in[idx] + bias[idx % depth];
}

// -------------------- ReLU --------------------
__global__ void gpu_relu_kernel(float *in, float *out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    out[idx] = fmaxf(0.0f, in[idx]);
}

// -------------------- Max Pooling (2x down) --------------------
__global__ void
gpu_max_pooling_kernel(float *in, float *out, int width, int height, int depth) {
  int x          = blockIdx.x * blockDim.x + threadIdx.x;
  int y          = blockIdx.y * blockDim.y + threadIdx.y;
  int d          = blockIdx.z * blockDim.z + threadIdx.z;
  int new_width  = width / 2;
  int new_height = height / 2;

  if (x >= new_width || y >= new_height || d >= depth)
    return;

  int in_x = x * 2;
  int in_y = y * 2;

  out[GET_1D_IDX(y, x, d, new_width, depth)] =
      fmaxf(fmaxf(in[GET_1D_IDX(in_y, in_x, d, width, depth)],
                  in[GET_1D_IDX(in_y, in_x + 1, d, width, depth)]),
            fmaxf(in[GET_1D_IDX(in_y + 1, in_x, d, width, depth)],
                  in[GET_1D_IDX(in_y + 1, in_x + 1, d, width, depth)]));
}

// -------------------- Upsampling (2x up) --------------------
__global__ void
gpu_upsampling_kernel(float *in, float *out, int width, int height, int depth) {
  int x          = blockIdx.x * blockDim.x + threadIdx.x;
  int y          = blockIdx.y * blockDim.y + threadIdx.y;
  int d          = blockIdx.z * blockDim.z + threadIdx.z;
  int new_width  = width * 2;
  int new_height = height * 2;

  if (x >= new_width || y >= new_height || d >= depth)
    return;

  int   out_x = 2 * x;
  int   out_y = 2 * y;
  float val   = in[GET_1D_IDX(y, x, d, width, depth)];

  out[GET_1D_IDX(out_y, out_x, d, new_width, depth)]         = val;
  out[GET_1D_IDX(out_y, out_x + 1, d, new_width, depth)]     = val;
  out[GET_1D_IDX(out_y + 1, out_x, d, new_width, depth)]     = val;
  out[GET_1D_IDX(out_y + 1, out_x + 1, d, new_width, depth)] = val;
}

// -------------------- Upsampling Backward --------------------
__global__ void gpu_upsampling_backward_kernel(
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

  d_sum += d_out[GET_1D_IDX(out_y, out_x, d, new_width, depth)];
  d_sum += d_out[GET_1D_IDX(out_y, out_x + 1, d, new_width, depth)];
  d_sum += d_out[GET_1D_IDX(out_y + 1, out_x, d, new_width, depth)];
  d_sum += d_out[GET_1D_IDX(out_y + 1, out_x + 1, d, new_width, depth)];

  d_in[GET_1D_IDX(y, x, d, width, depth)] = d_sum;
}

// -------------------- MSE Loss & Gradient --------------------
__global__ void
gpu_mse_loss_kernel(float *expected, float *actual, float *out, int size) {
  __shared__ float shared[MAX_BLOCKSIZE];

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
gpu_mse_grad_kernel(float *expected, float *actual, float *d_out, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    d_out[idx] = 2.0f * (actual[idx] - expected[idx]) / size;
}

// -------------------- ReLU Backward --------------------
__global__ void
gpu_relu_backward_kernel(float *in, float *d_out, float *d_in, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    d_out[idx] = in[idx] > 0 ? d_in[idx] : 0;
}

// -------------------- Bias Gradient --------------------
__global__ void gpu_bias_grad_kernel(float *d_out, float *d_bias, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    atomicAdd(d_out[idx], d_bias[idx % depth]);
}

// -------------------- Weight Update --------------------
__global__ void gpu_update_weight_kernel(float *weight,
                                         float *gradient,
                                         int    size,
                                         float  learning_rate) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    weight[idx] -= lr * gradient[idx];
}

// -------------------- Conv2D Backward: Filter Gradient only --------------------
__global__ void gpu_conv2D_grad_kernel(float *in,
                                       float *d_out,
                                       float *d_filter,
                                       int    width,
                                       int    height,
                                       int    depth,
                                       int    n_filter) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int f = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= width || y >= height || f >= n_filter)
    return;

  int   padding_x = CONV_FILTER_WIDTH / 2;
  int   padding_y = CONV_FILTER_HEIGHT / 2;
  float d_out_val = d_out[GET_1D_INDEX(i, j, f, width, n_filter)];

  for (int f_i = 0; f_i < CONV_FILTER_HEIGHT; ++f_i) {
    // If the row needs padding, we skip since we pad with 0
    int row = y + f_i - CONV_FILTER_HEIGHT / 2;
    if (row < 0 || row >= height)
      continue;

    for (int f_j = 0; f_j < CONV_FILTER_WIDTH; ++f_j) {
      // Same with column
      int col = x + f_j - CONV_FILTER_WIDTH / 2;
      if (col < 0 || col >= width)
        continue;

      // Calculate start of filter
      float *d_filter_start =
          d_filter + f * GET_1D_INDEX(f_i, f_j, 0, CONV_FILTER_WIDTH, depth);

      // Calculate start of input
      float *in_start = in + GET_1D_INDEX(row, col, 0, width, depth);
      for (int d = 0; d < depth; ++d)
        atomicAdd(d_filter_start + d, in_start[d] * d_out_val);
    }
  }
}

// GPU Max Pooling Backward
__global__ void gpu_max_pooling_backward_kernel(
    float *in, float *d_out, float *d_in, int width, int height, int depth) {
  int x     = blockIdx.x * blockDim.x + threadIdx.x;
  int y     = blockIdx.y * blockDim.y + threadIdx.y;
  int d     = blockIdx.z * blockDim.z + threadIdx.z;
  int out_x = x / 2;
  int out_y = y / 2;

  if (x >= width || y >= height || d >= depth)
    return;

  int    idx = GET_1D_IDX(y, x, d, width, depth);
  float *out = d_in + idx;
  float  val = d_out[GET_1D_IDX(out_y, out_x, d, width / 2, depth)];

  int neighbors_idx[]  = { GET_1D_INDEX(out_y * 2, out_x * 2, d, width, depth),
                           GET_1D_INDEX(out_y * 2, out_x * 2 + 1, d, width, depth),
                           GET_1D_INDEX(out_y * 2 + 1, out_x * 2, d, width, depth),
                           GET_1D_INDEX(out_y * 2 + 1, out_x * 2 + 1, d, width, depth) };
  int max_neighbor_idx = *max_element(
      neighbors_idx, neighbors_idx + 4, [in](int a, int b) { return in[a] < in[b]; });

  *out = (idx == max_neighbor_idx) ? val : 0.0f;
}

void gpu_conv2D(float *in,
                float *filter,
                float *out,
                int    n,
                int    width,
                int    height,
                int    depth,
                int    n_filter,
                dim3   block_size) {
  dim3 grid_size((block_size.x + width - 1) / block_size.x + 1,
                 (block_size.y + height - 1) / block_size.y + 1,
                 (block_size.z + n_filter - 1) / block_size.z + 1);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * n_filter;

    gpu_conv2D_kernel<<<grid_size, block_size>>>(
        in + in_offset, filter, out + out_offset, width, height, depth, n_filter);

    CUDA_CHECK(cudaGetLastError());
  }
}

void gpu_add_bias(float *in,
                  float *bias,
                  float *out,
                  int    n,
                  int    width,
                  int    height,
                  int    depth,
                  dim3   block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((block_size.x + size - 1) / block_size.x + 1);

  gpu_add_bias_kernel<<<grid_size, block_size>>>(in, bias, out, size);
  CUDA_CHECK(cudaGetLastError());
}

void gpu_relu(
    float *in, float *out, int n, int width, int height, int depth, dim3 block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((block_size.x + size - 1) / block_size.x + 1);

  gpu_relu_kernel<<<grid_size, block_size>>>(in, out, size);
  CUDA_CHECK(cudaGetLastError());
}

void gpu_max_pooling(
    float *in, float *out, int n, int width, int height, int depth, dim3 block_size) {
  dim3 grid_size((block_size.x + width - 1) / block_size.x + 1,
                 (block_size.y + height - 1) / block_size.y + 1,
                 (block_size.z + depth - 1) / block_size.z + 1);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * depth / 4;

    gpu_max_pooling_kernel<<<grid_size, block_size>>>(
        in + in_offset, out + out_offset, width, height, depth);

    CUDA_CHECK(cudaGetLastError());
  }
}

void gpu_upsampling(
    float *in, float *out, int n, int width, int height, int depth, dim3 block_size) {
  dim3 grid_size((block_size.x + width - 1) / block_size.x + 1,
                 (block_size.y + height - 1) / block_size.y + 1,
                 (block_size.z + depth - 1) / block_size.z + 1);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * depth * 4;

    gpu_upsampling_kernel<<<grid_size, block_size>>>(
        in + in_offset, out + out_offset, width, height, depth);

    CUDA_CHECK(cudaGetLastError());
  }
}

void gpu_mse_loss(float *expected,
                  float *actual,
                  float *mse,
                  int    n,
                  int    width,
                  int    height,
                  int    depth,
                  dim3   block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((block_size.x + size - 1) / block_size.x + 1);

  gpu_mse_loss_kernel<<<grid_size, block_size>>>(expected, actual, mse, size);
  CUDA_CHECK(cudaGetLastError());
}

void gpu_mse_grad(float *expected,
                  float *actual,
                  float *d_out,
                  int    n,
                  int    width,
                  int    height,
                  int    depth,
                  dim3   block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((block_size.x + size - 1) / block_size.x + 1);

  gpu_mse_grad_kernel<<<grid_size, block_size>>>(expected, actual, d_out, size);
  CUDA_CHECK(cudaGetLastError());
}

void gpu_relu_backward(float *in,
                       float *d_out,
                       float *d_in,
                       int    n,
                       int    width,
                       int    height,
                       int    depth,
                       dim3   block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((block_size.x + size - 1) / block_size.x + 1);

  gpu_relu_backward_kernel<<<grid_size, block_size>>>(in, d_out, d_in, size);
  CUDA_CHECK(cudaGetLastError());
}

void gpu_max_pooling_backward(float *in,
                              float *d_out,
                              float *d_in,
                              int    n,
                              int    width,
                              int    height,
                              int    depth,
                              dim3   block_size) {
  dim3 grid_size((block_size.x + width - 1) / block_size.x + 1,
                 (block_size.y + height - 1) / block_size.y + 1,
                 (block_size.z + depth - 1) / block_size.z + 1);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * depth / 4;

    gpu_max_pooling_backward_kernel<<<grid_size, block_size>>>(
        in + in_offset, d_out + out_offset, d_in + in_offset, width, height, depth);

    CUDA_CHECK(cudaGetLastError());
  }
}

void gpu_upsampling_backward(float *d_out,
                             float *d_in,
                             int    n,
                             int    width,
                             int    height,
                             int    depth,
                             dim3   block_size) {
  dim3 grid_size((block_size.x + width - 1) / block_size.x + 1,
                 (block_size.y + height - 1) / block_size.y + 1,
                 (block_size.z + depth - 1) / block_size.z + 1);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * depth * 4;

    gpu_upsampling_backward_kernel<<<grid_size, block_size>>>(
        d_out + out_offset, d_in + in_offset, width, height, depth);

    CUDA_CHECK(cudaGetLastError());
  }
}

void gpu_bias_grad(float *d_out,
                   float *d_bias,
                   int    n,
                   int    width,
                   int    height,
                   int    depth,
                   dim3   block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((block_size.x + size - 1) / block_size.x + 1);

  gpu_bias_grad_kernel<<<grid_size, block_size>>>(d_out, d_bias, size);
  CUDA_CHECK(cudaGetLastError());
}

void gpu_conv2D_grad(float *in,
                     float *d_out,
                     float *d_filter,
                     int    n,
                     int    width,
                     int    height,
                     int    depth,
                     int    n_filter,
                     dim3   block_size) {
  dim3 grid_size((block_size.x + width - 1) / block_size.x + 1,
                 (block_size.y + height - 1) / block_size.y + 1,
                 (block_size.z + depth - 1) / block_size.z + 1);

  for (int i = 0; i < n; ++i) {
    int in_offset  = i * width * height * depth;
    int out_offset = i * width * height * n_filter;

    gpu_conv2D_grad_kernel<<<grid_size, block_size>>>(
        in + in_offset, d_out + out_offset, d_in + in_offset, width, height, depth);

    CUDA_CHECK(cudaGetLastError());
  }
}

void gpu_update_weight(
    float *weight, float *gradient, int size, float learning_rate, dim3 block_size) {
  int  size = n * width * height * depth;
  dim3 grid_size((block_size.x + size - 1) / block_size.x + 1);

  gpu_update_weight_kernel<<<grid_size, block_size>>>(weight, gradient, size);
  CUDA_CHECK(cudaGetLastError());
}