#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include "constants.h"
#include "gpu_layers.h"

// Macro kiểm tra lỗi CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 1D index macro
#define GET_1D_INDEX(i,j,k,width,depth) ((k) + (depth) * ((j) + (i)*(width)))

// Threads per block
const int THREADS_PER_BLOCK = 256;

// ---------------------- KERNELS -------------------------

__global__ void conv2D_kernel(const float *d_in, const float *d_filter, float *d_out,
                              int n, int width, int height, int depth, int n_filter) {
    int image = blockIdx.x;
    int i = blockIdx.y;
    int j = blockIdx.z;
    int f = threadIdx.x;
    if (image >= n || i >= height || j >= width || f >= n_filter) return;
    int out_idx = image * width * height * n_filter + i * width * n_filter + j * n_filter + f;
    float sum = 0.0f;
    for(int fi=0; fi<CONV_FILTER_HEIGHT; ++fi){
        int row = i + fi - CONV_FILTER_HEIGHT/2;
        if(row<0 || row>=height) continue;
        for(int fj=0; fj<CONV_FILTER_WIDTH; ++fj){
            int col = j + fj - CONV_FILTER_WIDTH/2;
            if(col<0 || col>=width) continue;
            for(int d=0; d<depth; ++d){
                int in_idx = image*width*height*depth + row*width*depth + col*depth + d;
                int filter_idx = f*CONV_FILTER_HEIGHT*CONV_FILTER_WIDTH*depth + fi*CONV_FILTER_WIDTH*depth + fj*depth + d;
                sum += d_in[in_idx]*d_filter[filter_idx];
            }
        }
    }
    d_out[out_idx] = sum;
}

__global__ void conv2D_bias_relu_fwd_kernel(const float *d_in, const float *d_filter, const float *d_bias, float *d_out,
                                            int n, int width, int height, int depth, int n_filter) {
    int image = blockIdx.x;
    int i = blockIdx.y;
    int j = blockIdx.z;
    int f = threadIdx.x;
    if(image>=n || i>=height || j>=width || f>=n_filter) return;
    int out_idx = image * width * height * n_filter + i*width*n_filter + j*n_filter + f;
    float sum = 0.0f;
    for(int fi=0; fi<CONV_FILTER_HEIGHT; ++fi){
        int row = i+fi - CONV_FILTER_HEIGHT/2;
        if(row<0 || row>=height) continue;
        for(int fj=0; fj<CONV_FILTER_WIDTH; ++fj){
            int col = j+fj - CONV_FILTER_WIDTH/2;
            if(col<0 || col>=width) continue;
            for(int d=0; d<depth; ++d){
                int in_idx = image*width*height*depth + row*width*depth + col*depth + d;
                int filter_idx = f*CONV_FILTER_HEIGHT*CONV_FILTER_WIDTH*depth + fi*CONV_FILTER_WIDTH*depth + fj*depth + d;
                sum += d_in[in_idx]*d_filter[filter_idx];
            }
        }
    }
    sum += d_bias[f];
    d_out[out_idx] = fmaxf(sum,0.0f);
}

__global__ void add_bias_kernel(const float *d_in, const float *d_bias, float *d_out, int total_size, int depth){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=total_size) return;
    int d = idx % depth;
    d_out[idx] = d_in[idx] + d_bias[d];
}

__global__ void relu_kernel(const float *d_in, float *d_out, int total_size){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=total_size) return;
    d_out[idx] = fmaxf(d_in[idx],0.0f);
}

__global__ void max_pooling_kernel(const float *d_in, float *d_out, int n,int width,int height,int depth){
    int out_i = blockIdx.x;
    int out_j = blockIdx.y;
    int image = blockIdx.z;
    int out_width = width/2;
    int out_height = height/2;
    if(image>=n || out_i>=out_height || out_j>=out_width) return;
    int base_out = image*out_height*out_width*depth + out_i*out_width*depth + out_j*depth;
    int neighbors[4];
    neighbors[0]= image*width*height*depth + (out_i*2*width + out_j*2)*depth;
    neighbors[1]= neighbors[0]+depth;
    neighbors[2]= neighbors[0]+width*depth;
    neighbors[3]= neighbors[2]+depth;
    int k = threadIdx.x;
    if(k>=depth) return;
    float max_val = d_in[neighbors[0]+k];
    for(int ni=1;ni<4;ni++) max_val = fmaxf(max_val,d_in[neighbors[ni]+k]);
    d_out[base_out+k] = max_val;
}

__global__ void upsampling_kernel(const float *d_in, float *d_out,int n,int width,int height,int depth){
    int i = blockIdx.x;
    int j = blockIdx.y;
    int image = blockIdx.z;
    if(image>=n || i>=height || j>=width) return;
    int in_base = image*width*height*depth + i*width*depth + j*depth;
    int out_width = 2*width;
    int neighbors[4];
    neighbors[0]= image*out_width*2*depth + (i*2*out_width + j*2)*depth;
    neighbors[1]= neighbors[0]+depth;
    neighbors[2]= neighbors[0]+out_width*depth;
    neighbors[3]= neighbors[2]+depth;
    int k = threadIdx.x;
    if(k>=depth) return;
    float val = d_in[in_base+k];
    for(int ni=0;ni<4;ni++) d_out[neighbors[ni]+k]=val;
}

__global__ void mse_loss_kernel(const float *d_expected,const float *d_actual,float *d_out,int total_size){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=total_size) return;
    float diff = d_expected[idx]-d_actual[idx];
    d_out[idx] = diff*diff;
}

__global__ void mse_grad_kernel(const float *d_expected,const float *d_actual,float *d_grad,int total_size){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=total_size) return;
    d_grad[idx]=2.0f*(d_actual[idx]-d_expected[idx])/total_size;
}

__global__ void relu_backward_kernel(const float *d_in,const float *d_in_grad,float *d_out_grad,int total_size){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=total_size) return;
    d_out_grad[idx]= (d_in[idx]>0.0f)? d_in_grad[idx]:0.0f;
}

__global__ void bias_grad_kernel(const float *d_in_grad,float *d_bias_grad,int n_total_pixels,int depth){
    int d = blockIdx.x*blockDim.x + threadIdx.x;
    if(d>=depth) return;
    float sum=0.0f;
    for(int i=0;i<n_total_pixels;i++){
        sum += d_in_grad[i*depth + d];
    }
    d_bias_grad[d]=sum;
}

__global__ void conv2D_grad_kernel(const float *d_in,const float *d_out_grad,float *d_filter_grad,int n,int width,int height,int depth,int n_filter){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int total = n*width*height*n_filter;
    if(idx>=total) return;

    int image = idx / (width*height*n_filter);
    int rem = idx % (width*height*n_filter);
    int f = rem % n_filter;
    int pixel = rem / n_filter;
    int i = pixel / width;
    int j = pixel % width;

    float dout = d_out_grad[idx];
    for(int fi=0;fi<CONV_FILTER_HEIGHT;fi++){
        int row = i+fi - CONV_FILTER_HEIGHT/2;
        if(row<0 || row>=height) continue;
        for(int fj=0;fj<CONV_FILTER_WIDTH;fj++){
            int col = j+fj - CONV_FILTER_WIDTH/2;
            if(col<0 || col>=width) continue;
            for(int d=0;d<depth;d++){
                int in_idx = image*width*height*depth + row*width*depth + col*depth + d;
                int filter_idx = f*CONV_FILTER_HEIGHT*CONV_FILTER_WIDTH*depth + fi*CONV_FILTER_WIDTH*depth + fj*depth + d;
                atomicAdd(&d_filter_grad[filter_idx], d_in[in_idx]*dout);
            }
        }
    }
}

__global__ void update_weight_kernel(float *d_weight,const float *d_gradient,int size,float lr){
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=size) return;
    d_weight[idx]-= lr*d_gradient[idx];
}

// ------------------- HOST WRAPPERS -----------------------

void gpu_conv2D(float *d_in, float *d_filter, float *d_out,
                int n,int width,int height,int depth,int n_filter,cudaStream_t stream){
    dim3 grid(n,height,width);
    dim3 block(n_filter,1,1);
    conv2D_kernel<<<grid,block,0,stream>>>(d_in,d_filter,d_out,n,width,height,depth,n_filter);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_conv2D_bias_relu_fwd(float *d_in,float *d_filter,float *d_bias,float *d_out,
                              int n,int width,int height,int depth,int n_filter,cudaStream_t stream){
    dim3 grid(n,height,width);
    dim3 block(n_filter,1,1);
    conv2D_bias_relu_fwd_kernel<<<grid,block,0,stream>>>(d_in,d_filter,d_bias,d_out,n,width,height,depth,n_filter);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_add_bias(float *d_in,float *d_bias,float *d_out,int n,int width,int height,int depth,cudaStream_t stream){
    int total = n*width*height*depth;
    int blocks = (total + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    add_bias_kernel<<<blocks,THREADS_PER_BLOCK,0,stream>>>(d_in,d_bias,d_out,total,depth);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_relu(float *d_in,float *d_out,int n,int width,int height,int depth,cudaStream_t stream){
    int total = n*width*height*depth;
    int blocks = (total + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    relu_kernel<<<blocks,THREADS_PER_BLOCK,0,stream>>>(d_in,d_out,total);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_max_pooling(float *d_in,float *d_out,int n,int width,int height,int depth,cudaStream_t stream){
    int out_w = width/2;
    int out_h = height/2;
    dim3 grid(out_h,out_w,n);
    dim3 block(depth,1,1);
    max_pooling_kernel<<<grid,block,0,stream>>>(d_in,d_out,n,width,height,depth);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_upsampling(float *d_in,float *d_out,int n,int width,int height,int depth,cudaStream_t stream){
    dim3 grid(height,width,n);
    dim3 block(depth,1,1);
    upsampling_kernel<<<grid,block,0,stream>>>(d_in,d_out,n,width,height,depth);
    CUDA_CHECK(cudaGetLastError());
}

float gpu_mse_loss(float *d_expected,float *d_actual,int n,int width,int height,int depth,cudaStream_t stream){
    int total = n*width*height*depth;
    if(total==0) return 0.0f;
    float *d_sq;
    float *d_sum;
    int blocks = (total+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    CUDA_CHECK(cudaMalloc(&d_sq,total*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum,blocks*sizeof(float)));
    mse_loss_kernel<<<blocks,THREADS_PER_BLOCK,0,stream>>>(d_expected,d_actual,d_sq,total);
    CUDA_CHECK(cudaGetLastError());
    // simple reduction on CPU (safe)
    float *h_sq = new float[total];
    CUDA_CHECK(cudaMemcpy(h_sq,d_sq,total*sizeof(float),cudaMemcpyDeviceToHost));
    float sum=0.0f;
    for(int i=0;i<total;i++) sum+=h_sq[i];
    delete[] h_sq;
    cudaFree(d_sq);
    cudaFree(d_sum);
    return sum/total;
}

void gpu_mse_grad(float *d_expected,float *d_actual,float *d_out,int n,int width,int height,int depth,cudaStream_t stream){
    int total = n*width*height*depth;
    int blocks = (total + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    mse_grad_kernel<<<blocks,THREADS_PER_BLOCK,0,stream>>>(d_expected,d_actual,d_out,total);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_relu_backward(float *d_in,float *d_in_grad,float *d_out_grad,int n,int width,int height,int depth,cudaStream_t stream){
    int total = n*width*height*depth;
    int blocks = (total+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    relu_backward_kernel<<<blocks,THREADS_PER_BLOCK,0,stream>>>(d_in,d_in_grad,d_out_grad,total);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_bias_grad(float *d_in_grad,float *d_bias_grad,int n,int width,int height,int depth,cudaStream_t stream){
    int n_pixels = n*width*height;
    int blocks = (depth+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    bias_grad_kernel<<<blocks,THREADS_PER_BLOCK,0,stream>>>(d_in_grad,d_bias_grad,n_pixels,depth);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_conv2D_grad(float *d_in,float *d_out_grad,float *d_filter_grad,int n,int width,int height,int depth,int n_filter,cudaStream_t stream){
    int total = n*width*height*n_filter;
    int blocks = (total + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    conv2D_grad_kernel<<<blocks,THREADS_PER_BLOCK,0,stream>>>(d_in,d_out_grad,d_filter_grad,n,width,height,depth,n_filter);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_update_weight(float *d_weight,float *d_gradient,int size,float lr,cudaStream_t stream){
    int blocks = (size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    update_weight_kernel<<<blocks,THREADS_PER_BLOCK,0,stream>>>(d_weight,d_gradient,size,lr);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_load_parameter(float *d_param,const float *h_param,int size,cudaStream_t stream){
    CUDA_CHECK(cudaMemcpyAsync(d_param,h_param,size*sizeof(float),cudaMemcpyHostToDevice,stream));
}
