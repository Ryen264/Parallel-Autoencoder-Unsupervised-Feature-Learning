#include "constants.h"
#include "gpu_layers.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// -------------------- Conv2D Forward --------------------
__global__ void conv2D_kernel(const float *in, const float *filter, float *out,
                              int width, int height, int depth, int n_filter) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    for (int f = 0; f < n_filter; ++f) {
        int half_k = 1;
        float sum = 0.0f;
        for (int c = 0; c < depth; ++c) {
            for (int ky = -half_k; ky <= half_k; ++ky) {
                for (int kx = -half_k; kx <= half_k; ++kx) {
                    int ix = x + kx;
                    int iy = y + ky;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        int in_idx = c * width * height + iy * width + ix;
                        int filt_idx = f * depth * 3 * 3 + c * 3 * 3 + (ky + 1) * 3 + (kx + 1);
                        sum += in[in_idx] * filter[filt_idx];
                    }
                }
            }
        }
        out[f * width * height + y * width + x] = sum;
    }
}

void gpu_conv2D(float *in, float *filter, float *out,
                int n, int width, int height, int depth, int n_filter) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    for (int i = 0; i < n; ++i) {
        conv2D_kernel<<<grid, block>>>(in + i * width * height * depth,
                                       filter,
                                       out + i * width * height * n_filter,
                                       width, height, depth, n_filter);
        CUDA_CHECK(cudaGetLastError());
    }
}

// -------------------- Bias --------------------
__global__ void add_bias_kernel(float *in, const float *bias, float *out,
                                int width, int height, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height * depth;
    if (idx >= size) return;

    int c = (idx / (width * height)) % depth;
    out[idx] = in[idx] + bias[c];
}

void gpu_add_bias(float *in, float *bias, float *out,
                  int n, int width, int height, int depth) {
    int size = width * height * depth;
    int block = 256;
    int grid = (size + block - 1) / block;
    for (int i = 0; i < n; ++i) {
        add_bias_kernel<<<grid, block>>>(in + i * size, bias, out + i * size, width, height, depth);
        CUDA_CHECK(cudaGetLastError());
    }
}

// -------------------- ReLU --------------------
__global__ void relu_kernel(const float *in, float *out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = fmaxf(0.0f, in[idx]);
}

void gpu_relu(float *in, float *out, int n, int width, int height, int depth) {
    int size = n * width * height * depth;
    int block = 256;
    int grid = (size + block - 1) / block;
    relu_kernel<<<grid, block>>>(in, out, size);
    CUDA_CHECK(cudaGetLastError());
}

// -------------------- Max Pooling (2x down) --------------------
__global__ void max_pooling_kernel(const float *in, float *out,
                                   int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width/2 || y >= height/2) return;

    for (int c = 0; c < depth; ++c) {
        int in_x = x*2, in_y = y*2;
        float v0 = in[c*width*height + in_y*width + in_x];
        float v1 = in[c*width*height + in_y*width + in_x+1];
        float v2 = in[c*width*height + (in_y+1)*width + in_x];
        float v3 = in[c*width*height + (in_y+1)*width + in_x+1];
        out[c*(width/2)*(height/2) + y*(width/2) + x] = fmaxf(fmaxf(v0,v1), fmaxf(v2,v3));
    }
}

void gpu_max_pooling(float *in, float *out, int n, int width, int height, int depth) {
    dim3 block(16,16);
    dim3 grid((width/2 + 15)/16, (height/2 + 15)/16);
    for(int i=0;i<n;i++){
        max_pooling_kernel<<<grid, block>>>(in+i*width*height*depth,
                                            out+i*(width/2)*(height/2)*depth,
                                            width, height, depth);
        CUDA_CHECK(cudaGetLastError());
    }
}

// -------------------- Upsampling (2x up) --------------------
__global__ void upsampling_kernel(const float *in, float *out,
                                  int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    for (int c = 0; c < depth; ++c) {
        out[c*width*height + y*width + x] = in[c*(width/2)*(height/2) + (y/2)*(width/2) + (x/2)];
    }
}

void gpu_upsampling(float *in, float *out, int n, int width, int height, int depth) {
    dim3 block(16,16);
    dim3 grid((width+15)/16,(height+15)/16);
    for(int i=0;i<n;i++){
        upsampling_kernel<<<grid,block>>>(in+i*(width/2)*(height/2)*depth,
                                          out+i*width*height*depth,
                                          width,height,depth);
        CUDA_CHECK(cudaGetLastError());
    }
}

// -------------------- Upsampling Backward --------------------
__global__ void upsampling_backward_kernel(const float *d_in, float *d_out,int width,int height,int depth){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x>=width || y>=height) return;

    for(int c=0;c<depth;c++){
        atomicAdd(&d_out[c*(width/2)*(height/2)+(y/2)*(width/2)+(x/2)], d_in[c*width*height+y*width+x]);
    }
}

void gpu_upsampling_backward(float *d_in,float *d_out,int n,int width,int height,int depth){
    dim3 block(16,16);
    dim3 grid((width+15)/16,(height+15)/16);
    for(int i=0;i<n;i++){
        upsampling_backward_kernel<<<grid,block>>>(d_in+i*width*height*depth,
                                                   d_out+i*(width/2)*(height/2)*depth,
                                                   width,height,depth);
        CUDA_CHECK(cudaGetLastError());
    }
}

// -------------------- MSE Loss & Gradient --------------------
__global__ void mse_loss_kernel(const float *expected, const float *actual, float *loss, int size){
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0.0f;
    if(tid < size) temp = (expected[tid] - actual[tid])*(expected[tid] - actual[tid]);
    cache[cacheIndex] = temp;
    __syncthreads();

    int i = blockDim.x/2;
    while(i !=0){
        if(cacheIndex<i) cache[cacheIndex]+=cache[cacheIndex+i];
        __syncthreads();
        i/=2;
    }
    if(cacheIndex==0) atomicAdd(loss,cache[0]);
}

float gpu_mse_loss(float *expected, float *actual, int n, int width, int height, int depth){
    int size = n*width*height*depth;
    float *d_loss;
    float h_loss = 0.0f;
    CUDA_CHECK(cudaMalloc(&d_loss,sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss,0,sizeof(float)));
    int block = 256;
    int grid = (size + block -1)/block;
    mse_loss_kernel<<<grid,block>>>(expected,actual,d_loss,size);
    CUDA_CHECK(cudaMemcpy(&h_loss,d_loss,sizeof(float),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss));
    return h_loss/size;
}

__global__ void mse_grad_kernel(const float *expected, const float *actual, float *d_out, int size){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) d_out[idx] = 2.0f*(actual[idx]-expected[idx])/size;
}

void gpu_mse_grad(float *expected, float *actual, float *d_out,
                  int n, int width, int height, int depth){
    int size = n*width*height*depth;
    int block = 256;
    int grid = (size + block-1)/block;
    mse_grad_kernel<<<grid,block>>>(expected,actual,d_out,size);
    CUDA_CHECK(cudaGetLastError());
}

// -------------------- ReLU Backward --------------------
__global__ void relu_backward_kernel(const float *in, const float *d_in, float *d_out, int size){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) d_out[idx] = in[idx]>0 ? d_in[idx] : 0;
}

void gpu_relu_backward(float *in, float *d_in, float *d_out,
                       int n,int width,int height,int depth){
    int size = n*width*height*depth;
    int block = 256;
    int grid = (size+block-1)/block;
    relu_backward_kernel<<<grid,block>>>(in,d_in,d_out,size);
    CUDA_CHECK(cudaGetLastError());
}

// -------------------- Bias Gradient --------------------
__global__ void bias_grad_kernel(const float *d_in, float *d_bias,int width,int height,int depth){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c<depth){
        float sum=0.0f;
        for(int y=0;y<height;y++)
            for(int x=0;x<width;x++)
                sum += d_in[c*width*height + y*width + x];
        atomicAdd(&d_bias[c], sum);
    }
}

void gpu_bias_grad(float *d_in,float *d_bias,int n,int width,int height,int depth){
    int block=256;
    int grid=(depth+block-1)/block;
    for(int i=0;i<n;i++){
        bias_grad_kernel<<<grid,block>>>(d_in+i*width*height*depth,d_bias,width,height,depth);
        CUDA_CHECK(cudaGetLastError());
    }
}

// -------------------- Weight Update --------------------
__global__ void update_weight_kernel(float *weight,const float *gradient,int size,float lr){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) weight[idx]-=lr*gradient[idx];
}

void gpu_update_weight(float *weight, float *gradient,int size,float learning_rate){
    int block=256;
    int grid=(size+block-1)/block;
    update_weight_kernel<<<grid,block>>>(weight,gradient,size,learning_rate);
    CUDA_CHECK(cudaGetLastError());
}
// -------------------- Conv2D Backward: Filter Gradient only --------------------
__global__ void conv2D_grad_filter_kernel(
    const float *in, const float *d_out, float *d_filter,
    int width, int height, int depth, int n_filter)
{
    // grid: (n_filter, depth)
    int f = blockIdx.x; // [0..n_filter)
    int c = blockIdx.y; // [0..depth)
    int ky = threadIdx.y; // 0..2
    int kx = threadIdx.x; // 0..2

    if (f >= n_filter || c >= depth || ky >= 3 || kx >= 3) return;

    float sum = 0.0f;
    int plane = width * height;

    // accumulate over spatial positions
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int out_idx = f * plane + y * width + x;
            int iy = y + ky - 1;
            int ix = x + kx - 1;
            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                int in_idx = c * plane + iy * width + ix;
                sum += in[in_idx] * d_out[out_idx];
            }
        }
    }

    int filt_idx = f * depth * 9 + c * 9 + ky * 3 + kx;
    atomicAdd(&d_filter[filt_idx], sum);
}

void gpu_conv2D_grad(float *in, float *d_out, float *d_filter,
                     int n, int width, int height, int depth, int n_filter) {
    // zero d_filter
    size_t filt_bytes = (size_t)n_filter * depth * 9 * sizeof(float);
    CUDA_CHECK(cudaMemset(d_filter, 0, filt_bytes));

    dim3 blockF(3, 3);
    dim3 gridF(n_filter, depth);

    size_t in_stride = (size_t)width * height * depth;
    size_t dout_stride = (size_t)width * height * n_filter;

    for (int i = 0; i < n; ++i) {
        const float *in_i = in + i * in_stride;
        const float *dout_i = d_out + i * dout_stride;
        conv2D_grad_filter_kernel<<<gridF, blockF>>>(in_i, dout_i, d_filter, width, height, depth, n_filter);
        CUDA_CHECK(cudaGetLastError());
    }
}
// GPU Max Pooling Backward KERNEL AND WRAPPER
__global__ void max_pooling_backward_kernel(
    const float *in, const float *d_in, float *d_out,
    int width, int height, int depth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int out_w = width / 2;
    int out_h = height / 2;

    if (x >= out_w || y >= out_h) return;

    for (int c = 0; c < depth; ++c) {
        int in_x = x * 2;
        int in_y = y * 2;

        int base = c * width * height;
        float v0 = in[base + in_y * width + in_x];
        float v1 = in[base + in_y * width + in_x + 1];
        float v2 = in[base + (in_y + 1) * width + in_x];
        float v3 = in[base + (in_y + 1) * width + in_x + 1];

        float maxv = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));

        float grad = d_in[c * out_w * out_h + y * out_w + x];

        // truyền gradient vào đúng vị trí lớn nhất
        d_out[base + in_y * width + in_x]         = (v0 == maxv) ? grad : 0.0f;
        d_out[base + in_y * width + in_x + 1]     = (v1 == maxv) ? grad : 0.0f;
        d_out[base + (in_y + 1) * width + in_x]   = (v2 == maxv) ? grad : 0.0f;
        d_out[base + (in_y + 1) * width + in_x+1] = (v3 == maxv) ? grad : 0.0f;
    }
}

void gpu_max_pooling_backward(float *in, float *d_in, float *d_out,
                              int n, int width, int height, int depth)
{
    dim3 block(16, 16);
    dim3 grid((width/2 + 15) / 16, (height/2 + 15) / 16);

    // clear output gradient
    CUDA_CHECK(cudaMemset(d_out, 0, n * width * height * depth * sizeof(float)));

    int in_size = width * height * depth;
    int dout_size = (width/2) * (height/2) * depth;

    for (int i = 0; i < n; ++i)
    {
        max_pooling_backward_kernel<<<grid, block>>>(
            in + i * in_size,
            d_in + i * dout_size,
            d_out + i * in_size,
            width, height, depth
        );
        CUDA_CHECK(cudaGetLastError());
    }
}
