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
    int f = blockIdx.z;

    if (x >= width || y >= height || f >= n_filter) return;

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

void gpu_conv2D(float *in, float *filter, float *out,
                int n, int width, int height, int depth, int n_filter) {
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16, n_filter);

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
        add_bias_kernel<<<grid, block>>>(in + i*size, bias, out + i*size, width, height, depth);
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
    int c = blockIdx.z;

    if (x >= width/2 || y >= height/2) return;

    int in_x = x*2, in_y = y*2;
    float v0 = in[c*width*height + in_y*width + in_x];
    float v1 = in[c*width*height + in_y*width + in_x+1];
    float v2 = in[c*width*height + (in_y+1)*width + in_x];
    float v3 = in[c*width*height + (in_y+1)*width + in_x+1];

    out[c*(width/2)*(height/2) + y*(width/2) + x] = fmaxf(fmaxf(v0,v1), fmaxf(v2,v3));
}

void gpu_max_pooling(float *in, float *out, int n, int width, int height, int depth) {
    dim3 block(16,16);
    dim3 grid((width/2 + 15)/16, (height/2 + 15)/16, depth);
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
    int c = blockIdx.z;

    if (x >= width || y >= height) return;

    out[c*width*height + y*width + x] = in[c*(width/2)*(height/2) + (y/2)*(width/2) + (x/2)];
}

void gpu_upsampling(float *in, float *out, int n, int width, int height, int depth) {
    dim3 block(16,16);
    dim3 grid((width+15)/16,(height+15)/16,depth);
    for(int i=0;i<n;i++){
        upsampling_kernel<<<grid,block>>>(in+i*(width/2)*(height/2)*depth,
                                          out+i*width*height*depth,
                                          width,height,depth);
        CUDA_CHECK(cudaGetLastError());
    }
}

// -------------------- MSE Loss --------------------
__global__ void mse_loss_kernel(const float *expected, const float *actual, float *loss, int size){
    __shared__ float cache[256];
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0f;
    if(tid < size){
        float diff = expected[tid] - actual[tid];
        temp = diff*diff;
    }
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

// -------------------- MSE Gradient --------------------
__global__ void mse_grad_kernel(const float *expected, const float *actual, float *d_out, int size){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size){
        d_out[idx] = 2.0f*(actual[idx]-expected[idx])/size;
    }
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
    if(idx<size){
        d_out[idx] = in[idx]>0 ? d_in[idx] : 0;
    }
}

void gpu_relu_backward(float *in, float *d_in, float *d_out,
                       int n,int width,int height,int depth){
    int size = n*width*height*depth;
    int block = 256;
    int grid = (size+block-1)/block;
    relu_backward_kernel<<<grid,block>>>(in,d_in,d_out,size);
    CUDA_CHECK(cudaGetLastError());
}

// -------------------- Max Pooling Backward --------------------
__global__ void max_pooling_backward_kernel(const float *in, const float *d_in, float *d_out,
                                            int width,int height,int depth){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if(x>=width/2 || y>=height/2) return;

    int in_x = x*2, in_y = y*2;
    int idx_out = c*width*height + in_y*width + in_x;
    float val = in[idx_out];
    int max_idx = 0;
    float max_val = val;

    float vals[4] = {in[idx_out], in[idx_out+1], in[idx_out+width], in[idx_out+width+1]};
    for(int i=1;i<4;i++){
        if(vals[i]>max_val){ max_val=vals[i]; max_idx=i;}
    }

    int dx[4]={0,1,0,1}, dy[4]={0,0,1,1};
    for(int i=0;i<4;i++){
        int idx = c*width*height + (in_y+dy[i])*width + (in_x+dx[i]);
        d_out[idx] = (i==max_idx)? d_in[c*(width/2)*(height/2)+y*(width/2)+x]:0;
    }
}

void gpu_max_pooling_backward(float *in,float *d_in,float *d_out,
                              int n,int width,int height,int depth){
    dim3 block(16,16);
    dim3 grid((width/2 + 15)/16,(height/2 +15)/16,depth);
    for(int i=0;i<n;i++){
        max_pooling_backward_kernel<<<grid,block>>>(in+i*width*height*depth,
                                                    d_in+i*(width/2)*(height/2)*depth,
                                                    d_out+i*width*height*depth,
                                                    width,height,depth);
        CUDA_CHECK(cudaGetLastError());
    }
}

// -------------------- Upsampling Backward --------------------
__global__ void upsampling_backward_kernel(const float *d_in, float *d_out,int width,int height,int depth){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    if(x>=width || y>=height) return;

    atomicAdd(&d_out[c*(width/2)*(height/2)+(y/2)*(width/2)+(x/2)], d_in[c*width*height+y*width+x]);
}

void gpu_upsampling_backward(float *d_in,float *d_out,int n,int width,int height,int depth){
    dim3 block(16,16);
    dim3 grid((width+15)/16,(height+15)/16,depth);
    for(int i=0;i<n;i++){
        upsampling_backward_kernel<<<grid,block>>>(d_in+i*width*height*depth,
                                                   d_out+i*(width/2)*(height/2)*depth,
                                                   width,height,depth);
        CUDA_CHECK(cudaGetLastError());
    }
}

// -------------------- Bias Gradient --------------------
__global__ void bias_grad_kernel(const float *d_in, float *d_bias,int width,int height,int depth){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c<depth){
        float sum=0.0f;
        for(int y=0;y<height;y++){
            for(int x=0;x<width;x++){
                sum += d_in[c*width*height + y*width + x];
            }
        }
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

#endif
