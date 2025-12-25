#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <cuda_runtime.h>
#include "constants.h"
#include "gpu_autoencoder.h"
#include "gpu_layers.h"

using std::max_element;
using std::printf, std::puts;
using std::stringstream;
using std::swap;

#define CUDA_CHECK(err)                                          \
    {                                                            \
        if (err != cudaSuccess)                                  \
        {                                                        \
            printf("\n\n!!! CRITICAL CUDA ERROR (File: %s, Line: %d) !!!\n", __FILE__, __LINE__); \
            printf("Code: %d, Message: %s\n", err, cudaGetErrorString(err)); \
            /* Thêm exit để dừng chương trình ngay khi phát hiện lỗi CUDA */ \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    }

Gpu_Autoencoder::Gpu_Autoencoder(): IAutoencoder() {};
Gpu_Autoencoder::Gpu_Autoencoder(const char *filename): IAutoencoder(filename) {};

// -----------------------------------------------------
// FORWARD PASS HELPERS (Fixed Dimensions)
// -----------------------------------------------------

Dataset Gpu_Autoencoder::_encode_save_output(const Dataset &dataset)
{
    int n = dataset.n, depth = dataset.depth;
    int w_curr = dataset.width;
    int h_curr = dataset.height;

    // First conv2D layer (Dim: n * W * H * F1)
    gpu_conv2D(dataset.get_data(), _encoder_filter_1.get(), _out_encoder_filter_1.get(),
               n, w_curr, h_curr, depth, _ENCODER_FILTER_1_DEPTH);

    // Dim: n * W * H * F1
    gpu_add_bias(_out_encoder_filter_1.get(), _encoder_bias_1.get(), _out_encoder_bias_1.get(),
                 n, w_curr, h_curr, _ENCODER_FILTER_1_DEPTH);

    // ReLU layer
    gpu_relu(_out_encoder_bias_1.get(), _out_encoder_relu_1.get(),
             n, w_curr, h_curr, _ENCODER_FILTER_1_DEPTH);

    // First max pooling layer (Output Dim: n * W/2 * H/2 * F1)
    int w_half = w_curr / 2;
    int h_half = h_curr / 2;
    gpu_max_pooling(_out_encoder_relu_1.get(), _out_max_pooling_1.get(),
                    n, w_curr, h_curr, _ENCODER_FILTER_1_DEPTH);
    w_curr = w_half;
    h_curr = h_half; // Update dimensions

    // Second conv2D layer (Dim: n * W/2 * H/2 * F2)
    gpu_conv2D(_out_max_pooling_1.get(), _encoder_filter_2.get(), _out_encoder_filter_2.get(),
               n, w_curr, h_curr, _ENCODER_FILTER_1_DEPTH, _ENCODER_FILTER_2_DEPTH);

    // Dim: n * W/2 * H/2 * F2
    gpu_add_bias(_out_encoder_filter_2.get(), _encoder_bias_2.get(), _out_encoder_bias_2.get(),
                 n, w_curr, h_curr, _ENCODER_FILTER_2_DEPTH);

    // ReLU layer
    gpu_relu(_out_encoder_bias_2.get(), _out_encoder_relu_2.get(),
             n, w_curr, h_curr, _ENCODER_FILTER_2_DEPTH);

    // Second max pooling layer (Output Dim: n * W/4 * H/4 * F2)
    int w_quarter = w_curr / 2;
    int h_quarter = h_curr / 2;
    gpu_max_pooling(_out_encoder_relu_2.get(), _out_max_pooling_2.get(),
                    n, w_curr, h_curr, _ENCODER_FILTER_2_DEPTH);
    w_curr = w_quarter;
    h_curr = h_quarter; // Update dimensions

    // Return the result (Dim: n * W/4 * H/4 * F2)
    int n_elements = n * w_curr * h_curr * _ENCODER_FILTER_2_DEPTH;
    Dataset res(n, w_curr, h_curr, _ENCODER_FILTER_2_DEPTH);
    cudaError_t err = cudaMemcpy(res.get_data(), _out_max_pooling_2.get(),
                                 n_elements * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    return res;
}

Dataset Gpu_Autoencoder::_decode_save_output(const Dataset &dataset)
{
    int n = dataset.n, depth = dataset.depth;
    int w_curr = dataset.width; // W/4
    int h_curr = dataset.height; // H/4

    // First conv2D layer (Output Dim: n * W/4 * H/4 * D1)
    gpu_conv2D(dataset.get_data(), _decoder_filter_1.get(), _out_decoder_filter_1.get(),
               n, w_curr, h_curr, depth, _DECODER_FILTER_1_DEPTH);

    // Dim: n * W/4 * H/4 * D1
    gpu_add_bias(_out_decoder_filter_1.get(), _decoder_bias_1.get(), _out_decoder_bias_1.get(),
                 n, w_curr, h_curr, _DECODER_FILTER_1_DEPTH);

    // ReLU layer
    gpu_relu(_out_decoder_bias_1.get(), _out_decoder_relu_1.get(),
             n, w_curr, h_curr, _DECODER_FILTER_1_DEPTH);

    // First upsampling layer (Output Dim: n * W/2 * H/2 * D1)
    w_curr *= 2; // W/2
    h_curr *= 2; // H/2
    gpu_upsampling(_out_decoder_relu_1.get(), _out_upsampling_1.get(),
                   n, w_curr, h_curr, _DECODER_FILTER_1_DEPTH);

    // Second conv2D layer (Output Dim: n * W/2 * H/2 * D2)
    gpu_conv2D(_out_upsampling_1.get(), _decoder_filter_2.get(), _out_decoder_filter_2.get(),
               n, w_curr, h_curr, _DECODER_FILTER_1_DEPTH, _DECODER_FILTER_2_DEPTH);

    // Dim: n * W/2 * H/2 * D2
    gpu_add_bias(_out_decoder_filter_2.get(), _decoder_bias_2.get(), _out_decoder_bias_2.get(),
                 n, w_curr, h_curr, _DECODER_FILTER_2_DEPTH);

    // ReLU layer
    gpu_relu(_out_decoder_bias_2.get(), _out_decoder_relu_2.get(),
             n, w_curr, h_curr, _DECODER_FILTER_2_DEPTH);

    // Second upsampling layer (Output Dim: n * W * H * D2)
    w_curr *= 2; // W
    h_curr *= 2; // H
    gpu_upsampling(_out_decoder_relu_2.get(), _out_upsampling_2.get(),
                   n, w_curr, h_curr, _DECODER_FILTER_2_DEPTH);

    // Third conv2D layer (Final Output Dim: n * W * H * D3)
    gpu_conv2D(_out_upsampling_2.get(), _decoder_filter_3.get(), _out_decoder_filter_3.get(),
               n, w_curr, h_curr, _DECODER_FILTER_2_DEPTH, _DECODER_FILTER_3_DEPTH);

    // Dim: n * W * H * D3
    gpu_add_bias(_out_decoder_filter_3.get(), _decoder_bias_3.get(), _out_decoder_bias_3.get(),
                 n, w_curr, h_curr, _DECODER_FILTER_3_DEPTH);

    // Return the result (Dim: n * W * H * D3)
    Dataset res(n, w_curr, h_curr, _DECODER_FILTER_3_DEPTH);
    int total_output_elements = n * w_curr * h_curr * _DECODER_FILTER_3_DEPTH;

    cudaError_t err = cudaMemcpy(res.get_data(), _out_decoder_bias_3.get(),
                                 total_output_elements * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    return res;
}

// -----------------------------------------------------
// MEMORY MANAGEMENT (No changes needed, fixed logic above ensures n_pixel_max_tensor is correct)
// -----------------------------------------------------

void Gpu_Autoencoder::_allocate_output_mem(int n, int width, int height)
{
    // Calculate different pixel counts for different resolution levels
    int n_pixel = n * width * height;                     // Original: n * w * h (32x32)
    int n_pixel_half = n * (width / 2) * (height / 2);    // After 1st pool: n * w/2 * h/2 (16x16)
    int n_pixel_quarter = n * (width / 4) * (height / 4); // After 2nd pool: n * w/4 * h/4 (8x8)

    // Max dimension for temporary buffers (d_in, d_out) is the output of the decoder (n * w * h * D3)
    int n_pixel_max_tensor = n_pixel; 
    int max_depth_intermediate = std::max({_DECODER_FILTER_3_DEPTH, _DECODER_FILTER_2_DEPTH, _ENCODER_FILTER_1_DEPTH});

    float *d_ptr = nullptr;

    // --- Encoder output allocations ---
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel * _ENCODER_FILTER_1_DEPTH * sizeof(float)));
    _out_encoder_filter_1 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                   { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel * _ENCODER_FILTER_1_DEPTH * sizeof(float)));
    _out_encoder_bias_1 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                 { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel * _ENCODER_FILTER_1_DEPTH * sizeof(float)));
    _out_encoder_relu_1 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                 { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_half * _ENCODER_FILTER_1_DEPTH * sizeof(float)));
    _out_max_pooling_1 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_half * _ENCODER_FILTER_2_DEPTH * sizeof(float)));
    _out_encoder_filter_2 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                   { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_half * _ENCODER_FILTER_2_DEPTH * sizeof(float)));
    _out_encoder_bias_2 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                 { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_half * _ENCODER_FILTER_2_DEPTH * sizeof(float)));
    _out_encoder_relu_2 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                 { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_quarter * _ENCODER_FILTER_2_DEPTH * sizeof(float)));
    _out_max_pooling_2 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                { cudaFree(p); });

    // --- Decoder output allocations ---
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_quarter * _DECODER_FILTER_1_DEPTH * sizeof(float)));
    _out_decoder_filter_1 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                   { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_quarter * _DECODER_FILTER_1_DEPTH * sizeof(float)));
    _out_decoder_bias_1 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                 { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_quarter * _DECODER_FILTER_1_DEPTH * sizeof(float)));
    _out_decoder_relu_1 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                 { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_half * _DECODER_FILTER_1_DEPTH * sizeof(float)));
    _out_upsampling_1 = std::shared_ptr<float>(d_ptr, [](float *p)
                                               { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_half * _DECODER_FILTER_2_DEPTH * sizeof(float)));
    _out_decoder_filter_2 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                   { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_half * _DECODER_FILTER_2_DEPTH * sizeof(float)));
    _out_decoder_bias_2 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                 { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_half * _DECODER_FILTER_2_DEPTH * sizeof(float)));
    _out_decoder_relu_2 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                 { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel * _DECODER_FILTER_2_DEPTH * sizeof(float)));
    _out_upsampling_2 = std::shared_ptr<float>(d_ptr, [](float *p)
                                               { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel * _DECODER_FILTER_3_DEPTH * sizeof(float)));
    _out_decoder_filter_3 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                   { cudaFree(p); });
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel * _DECODER_FILTER_3_DEPTH * sizeof(float)));
    _out_decoder_bias_3 = std::shared_ptr<float>(d_ptr, [](float *p)
                                                 { cudaFree(p); });

    // --- Allocate temporary buffers ---
    static constexpr int FILTER_SIZES[] = {_ENCODER_FILTER_1_SIZE,
                                           _ENCODER_FILTER_2_SIZE,
                                           _DECODER_FILTER_1_SIZE,
                                           _DECODER_FILTER_2_SIZE,
                                           _DECODER_FILTER_3_SIZE};
    constexpr int MAX_FILTER_SIZE = *max_element(FILTER_SIZES, FILTER_SIZES + 5);

    // Use n_pixel_max_tensor (n*W*H) and max_depth_intermediate (max depth, likely 256)
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_max_tensor * max_depth_intermediate * sizeof(float)));
    _d_in = std::shared_ptr<float>(d_ptr, [](float *p)
                                   { cudaFree(p); });

    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_max_tensor * max_depth_intermediate * sizeof(float)));
    _d_out = std::shared_ptr<float>(d_ptr, [](float *p)
                                    { cudaFree(p); });

    CUDA_CHECK(cudaMalloc(&d_ptr, MAX_FILTER_SIZE * sizeof(float)));
    _d_filter = std::shared_ptr<float>(d_ptr, [](float *p)
                                       { cudaFree(p); });
}

void Gpu_Autoencoder::_deallocate_output_mem()
{
    _out_encoder_filter_1 = nullptr;
    _out_encoder_bias_1 = nullptr;
    _out_encoder_relu_1 = nullptr;
    _out_max_pooling_1 = nullptr;

    _out_encoder_filter_2 = nullptr;
    _out_encoder_bias_2 = nullptr;
    _out_encoder_relu_2 = nullptr;
    _out_max_pooling_2 = nullptr;

    _out_decoder_filter_1 = nullptr;
    _out_decoder_bias_1 = nullptr;
    _out_decoder_relu_1 = nullptr;
    _out_upsampling_1 = nullptr;

    _out_decoder_filter_2 = nullptr;
    _out_decoder_bias_2 = nullptr;
    _out_decoder_relu_2 = nullptr;
    _out_upsampling_2 = nullptr;

    _out_decoder_filter_3 = nullptr;
    _out_decoder_bias_3 = nullptr;

    _d_in = nullptr;
    _d_out = nullptr;
    _d_filter = nullptr;
}

// -----------------------------------------------------
// TRAINING AND EVALUATION (Fixed Dimensions)
// -----------------------------------------------------

float Gpu_Autoencoder::_fit_batch(const Dataset &batch, float learning_rate)
{
    // Get dimensions of the input batch (W=32, H=32, D=3)
    int n = batch.n, width = batch.width, height = batch.height; 
    int depth_out = _DECODER_FILTER_3_DEPTH; // D3 = 3 (output depth)

    float *d_in = _d_in.get(), *d_out = _d_out.get(), *d_filter = _d_filter.get();
    Dataset encoded = _encode_save_output(batch); // Output: n * W/4 * H/4 * F2
    Dataset res = _decode_save_output(encoded);   // Output: n * W * H * D3

    // --------------------------------------------------------------------------------
    // FORWARD LOSS CALCULATION
    // --------------------------------------------------------------------------------
    float loss = gpu_mse_loss(batch.get_data(), res.get_data(),
                              n, width, height, depth_out);
    CUDA_CHECK(cudaGetLastError());

    // Get gradient of Loss w.r.t final output (d_out is W x H x D3 gradient)
    gpu_mse_grad(batch.get_data(), res.get_data(), d_out,
                 n, width, height, depth_out);
    CUDA_CHECK(cudaGetLastError());

    // --------------------------------------------------------------------------------
    // DECODER BACKWARD PASS (from W x H to W/4 x H/4)
    // --------------------------------------------------------------------------------

    // Bias 3 Grad (W x H x D3)
    gpu_bias_grad(d_out, d_in, n, width, height, depth_out);
    CUDA_CHECK(cudaGetLastError());

    gpu_update_weight(_decoder_bias_3.get(), d_in, depth_out, learning_rate);
    CUDA_CHECK(cudaGetLastError());

    // Conv 3 Grad (W x H) - Input to layer: _out_upsampling_2 (W x H x D2)
    // d_out size: n * W * H * D3
    gpu_conv2D_grad(_out_upsampling_2.get(), d_out, d_filter,
                    n, width, height, _DECODER_FILTER_2_DEPTH, depth_out);
    CUDA_CHECK(cudaGetLastError());

    // Conv 3 Backprop-to-Input (d_in is W x H x D2 gradient) - Sửa lỗi cấu trúc
    gpu_conv2D_backward_data(d_out, _decoder_filter_3.get(), d_in,
               n, width, height, depth_out, _DECODER_FILTER_2_DEPTH);
    CUDA_CHECK(cudaGetLastError());
    swap(d_out, d_in); // d_out is now W x H x D2 gradient

    gpu_update_weight(_decoder_filter_3.get(), d_filter, _DECODER_FILTER_3_SIZE, learning_rate);
    CUDA_CHECK(cudaGetLastError());

    // Upsampling 2 Backward (W x H -> W/2 x H/2)
    int w_half = width / 2; // 16
    int h_half = height / 2; // 16
    // d_out (W x H x D2) is the gradient, we pass original W, H
    gpu_upsampling_backward(d_out, d_in, n, width, height, _DECODER_FILTER_2_DEPTH);
    CUDA_CHECK(cudaGetLastError());
    swap(d_out, d_in); // d_out is now W/2 x H/2 x D2 gradient

    // ReLU 2 Backward (W/2 x H/2 x D2)
    gpu_relu_backward(_out_decoder_bias_2.get(), d_in, d_out,
                      n, w_half, h_half, _DECODER_FILTER_2_DEPTH);
    CUDA_CHECK(cudaGetLastError());
    swap(d_out, d_in); // d_out is now W/2 x H/2 x D2 gradient

    // Bias 2 Grad (W/2 x H/2 x D2)
    gpu_bias_grad(d_out, d_in, n, w_half, h_half, _DECODER_FILTER_2_DEPTH);
    CUDA_CHECK(cudaGetLastError());

    gpu_update_weight(_decoder_bias_2.get(), d_in, _DECODER_FILTER_2_DEPTH, learning_rate);
    CUDA_CHECK(cudaGetLastError());

    // Conv 2 Grad (W/2 x H/2) - Input to layer: _out_upsampling_1 (W/2 x H/2 x D1)
    // d_out size: n * W/2 * H/2 * D2
    gpu_conv2D_grad(_out_upsampling_1.get(), d_out, d_filter,
                    n, w_half, h_half, _DECODER_FILTER_1_DEPTH, _DECODER_FILTER_2_DEPTH);
    CUDA_CHECK(cudaGetLastError());

    // Conv 2 Backprop-to-Input (d_in is W/2 x H/2 x D1 gradient) - Sửa lỗi cấu trúc
    gpu_conv2D_backward_data(d_out, _decoder_filter_2.get(), d_in,
               n, w_half, h_half, _DECODER_FILTER_2_DEPTH, _DECODER_FILTER_1_DEPTH);
    CUDA_CHECK(cudaGetLastError());
    swap(d_out, d_in); // d_out is now W/2 x H/2 x D1 gradient

    gpu_update_weight(_decoder_filter_2.get(), d_filter, _DECODER_FILTER_2_SIZE, learning_rate);
    CUDA_CHECK(cudaGetLastError());

    // Upsampling 1 Backward (W/2 x H/2 -> W/4 x H/4)
    int w_quarter = width / 4; // 8
    int h_quarter = height / 4; // 8
    // d_out (W/2 x H/2 x D1) is the gradient, pass original W/2, H/2
    gpu_upsampling_backward(d_out, d_in, n, w_half, h_half, _DECODER_FILTER_1_DEPTH);
    CUDA_CHECK(cudaGetLastError());
    swap(d_out, d_in); // d_out is now W/4 x H/4 x D1 gradient

    // ReLU 1 Backward (W/4 x H/4 x D1)
    gpu_relu_backward(_out_decoder_bias_1.get(), d_in, d_out,
                      n, w_quarter, h_quarter, _DECODER_FILTER_1_DEPTH);
    CUDA_CHECK(cudaGetLastError());
    swap(d_out, d_in); // d_out is now W/4 x H/4 x D1 gradient

    // Bias 1 Grad (W/4 x H/4 x D1)
    gpu_bias_grad(d_out, d_in, n, w_quarter, h_quarter, _DECODER_FILTER_1_DEPTH);
    CUDA_CHECK(cudaGetLastError());

    gpu_update_weight(_decoder_bias_1.get(), d_in, _DECODER_FILTER_1_DEPTH, learning_rate);
    CUDA_CHECK(cudaGetLastError());

    // Conv 1 Grad (W/4 x H/4) - Input to layer: _out_max_pooling_2 (W/4 x H/4 x F2 - Latent Space)
    // d_out size: n * W/4 * H/4 * D1
    gpu_conv2D_grad(_out_max_pooling_2.get(), d_out, d_filter,
                    n, w_quarter, h_quarter, _ENCODER_FILTER_2_DEPTH, _DECODER_FILTER_1_DEPTH);
    CUDA_CHECK(cudaGetLastError());

    // Conv 1 Backprop-to-Input (d_in is W/4 x H/4 x F2 gradient - Latent Space) - Sửa lỗi cấu trúc
    gpu_conv2D_backward_data(d_out, _decoder_filter_1.get(), d_in,
               n, w_quarter, h_quarter, _DECODER_FILTER_1_DEPTH, _ENCODER_FILTER_2_DEPTH);
    CUDA_CHECK(cudaGetLastError());
    swap(d_out, d_in); // d_out is now W/4 x H/4 x F2 gradient (Latent Space Delta)

    gpu_update_weight(_decoder_filter_1.get(), d_filter, _DECODER_FILTER_1_SIZE, learning_rate);
    CUDA_CHECK(cudaGetLastError());

    // --------------------------------------------------------------------------------
    // ENCODER BACKWARD PASS (from W/4 x H/4 to W x H)
    // --------------------------------------------------------------------------------
    
    // Max pooling 2 Backward (W/4 x H/4 -> W/2 x H/2)
    // d_out (W/4 x H/4 x F2) is the gradient, pass original W/2, H/2
    gpu_max_pooling_backward(_out_encoder_relu_2.get(), d_out, d_in,
                             n, w_half, h_half, _ENCODER_FILTER_2_DEPTH);
    CUDA_CHECK(cudaGetLastError());
    swap(d_out, d_in); // d_out is now W/2 x H/2 x F2 gradient

    // ReLU 2 Backward (W/2 x H/2 x F2)
    gpu_relu_backward(_out_encoder_bias_2.get(), d_in, d_out,
                      n, w_half, h_half, _ENCODER_FILTER_2_DEPTH);
    CUDA_CHECK(cudaGetLastError());
    swap(d_out, d_in); // d_out is now W/2 x H/2 x F2 gradient

    // Bias 2 Grad (W/2 x H/2 x F2)
    gpu_bias_grad(d_out, d_in, n, w_half, h_half, _ENCODER_FILTER_2_DEPTH);
    CUDA_CHECK(cudaGetLastError());

    gpu_update_weight(_encoder_bias_2.get(), d_in, _ENCODER_FILTER_2_DEPTH, learning_rate);
    CUDA_CHECK(cudaGetLastError());

    // Conv 2 Grad (W/2 x H/2) - Input to layer: _out_max_pooling_1 (W/2 x H/2 x F1)
    // d_out size: n * W/2 * H/2 * F2
    gpu_conv2D_grad(_out_max_pooling_1.get(), d_out, d_filter,
                    n, w_half, h_half, _ENCODER_FILTER_1_DEPTH, _ENCODER_FILTER_2_DEPTH);
    CUDA_CHECK(cudaGetLastError());

    // Conv 2 Backprop-to-Input (d_in is W/2 x H/2 x F1 gradient) - Sửa lỗi cấu trúc
    gpu_conv2D_backward_data(d_out, _encoder_filter_2.get(), d_in,
               n, w_half, h_half, _ENCODER_FILTER_2_DEPTH, _ENCODER_FILTER_1_DEPTH);
    CUDA_CHECK(cudaGetLastError());
    swap(d_out, d_in); // d_out is now W/2 x H/2 x F1 gradient

    gpu_update_weight(_encoder_filter_2.get(), d_filter, _ENCODER_FILTER_2_SIZE, learning_rate);
    CUDA_CHECK(cudaGetLastError());

    // Max pooling 1 Backward (W/2 x H/2 -> W x H)
    // d_out (W/2 x H/2 x F1) is the gradient, pass original W, H
    gpu_max_pooling_backward(_out_encoder_relu_1.get(), d_out, d_in,
                             n, width, height, _ENCODER_FILTER_1_DEPTH);
    CUDA_CHECK(cudaGetLastError());
    swap(d_out, d_in); // d_out is now W x H x F1 gradient

    // ReLU 1 Backward (W x H x F1)
    gpu_relu_backward(_out_encoder_bias_1.get(), d_in, d_out,
                      n, width, height, _ENCODER_FILTER_1_DEPTH);
    CUDA_CHECK(cudaGetLastError());
    swap(d_out, d_in); // d_out is now W x H x F1 gradient

    // Bias 1 Grad (W x H x F1)
    gpu_bias_grad(d_out, d_in, n, width, height, _ENCODER_FILTER_1_DEPTH);
    CUDA_CHECK(cudaGetLastError());

    gpu_update_weight(_encoder_bias_1.get(), d_in, _ENCODER_FILTER_1_DEPTH, learning_rate);
    CUDA_CHECK(cudaGetLastError());

    // Conv 1 Grad (W x H) - Input to layer: batch.get_data() (W x H x D_input)
    // d_out size: n * W * H * F1
    gpu_conv2D_grad(batch.get_data(), d_out, d_filter,
                    n, width, height, batch.depth, _ENCODER_FILTER_1_DEPTH);
    CUDA_CHECK(cudaGetLastError());

    // Final output d_in: n * W * H * D_input (gradient to input, not used) - Sửa lỗi cấu trúc
    gpu_conv2D_backward_data(d_out, _encoder_filter_1.get(), d_in,
               n, width, height, _ENCODER_FILTER_1_DEPTH, batch.depth);
    CUDA_CHECK(cudaGetLastError());
    swap(d_out, d_in);

    gpu_update_weight(_encoder_filter_1.get(), d_filter, _ENCODER_FILTER_1_SIZE, learning_rate);
    CUDA_CHECK(cudaGetLastError());

    return loss;
}

void Gpu_Autoencoder::fit(const Dataset &dataset, int n_epoch, int batch_size, float learning_rate,
                          bool verbose, int checkpoint, const char *output_dir)
{
    // Create minibatches
    vector<Dataset> batches = create_minibatches(dataset, batch_size);

    // Allocate memory for training
    _allocate_output_mem(batch_size, dataset.width, dataset.height);

    puts("=======================TRAINING START=======================");
    for (int epoch = 1; epoch <= n_epoch; ++epoch)
    {
        printf("Epoch %d:\n", epoch);

        float total_loss = 0;
        for (const Dataset &batch : batches)
        {
            total_loss += _fit_batch(batch, learning_rate) * batch.n;
        }

        // Print average loss for the epoch
        float avg_loss = total_loss / dataset.n;
        printf("  Loss: %.4f\n", avg_loss);

        // Save at checkpoints
        if (checkpoint > 0 && epoch % checkpoint == 0)
        {
            stringstream builder;
            builder << output_dir << '/' << "autoencoder_" << epoch << ".bin";
            _save_paramters(builder.str().c_str());
        }
    }
    puts("========================TRAINING END========================");

    // Deallocate memory to remove unused memory
    _deallocate_output_mem();

    // Save models param
    stringstream builder;
    builder << output_dir << '/' << "autoencoder_.bin";
    _save_paramters(builder.str().c_str());
}

// -----------------------------------------------------
// ENCODE/DECODE (Batch Processing) (Fixed Dimensions)
// -----------------------------------------------------

Dataset Gpu_Autoencoder::encode(const Dataset &dataset) const
{
    // Encode by batches to use less memory
    int width = dataset.width, height = dataset.height, depth = dataset.depth;
    int w_curr = width;
    int h_curr = height;
    
    // Calculate final output dimensions
    int w_quarter = width / 4;
    int h_quarter = height / 4;
    int encoded_image_elements = w_quarter * h_quarter * _ENCODER_FILTER_2_DEPTH;
    int offset = 0; // Offset tính bằng số float (phần tử)

    vector<Dataset> batches = create_minibatches(dataset, _ENCODE_BATCH_SIZE);
    Dataset res(dataset.n, w_quarter, h_quarter, _ENCODER_FILTER_2_DEPTH);

    // Placeholder, alternating
    float *a, *b;
    // Max intermediate size is W x H (original size)
    int n_pixel_max = _ENCODE_BATCH_SIZE * width * height; 
    CUDA_CHECK(cudaMalloc(&a, n_pixel_max * MAX_FILTER_DEPTH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, n_pixel_max * MAX_FILTER_DEPTH * sizeof(float)));

    for (size_t i = 0; i < batches.size(); ++i)
    {
        int n_batch = batches[i].n;
        w_curr = width;
        h_curr = height;

        // First conv2D (size W x H)
        gpu_conv2D(batches[i].get_data(), _encoder_filter_1.get(), a,
                   n_batch, w_curr, h_curr, depth, _ENCODER_FILTER_1_DEPTH);

        // Add bias
        gpu_add_bias(a, _encoder_bias_1.get(), b,
                     n_batch, w_curr, h_curr, _ENCODER_FILTER_1_DEPTH);

        // ReLU
        gpu_relu(b, a, n_batch, w_curr, h_curr, _ENCODER_FILTER_1_DEPTH);

        // Max pooling 1 (size W/2 x H/2)
        gpu_max_pooling(a, b, n_batch, w_curr, h_curr, _ENCODER_FILTER_1_DEPTH);
        w_curr = width / 2;
        h_curr = height / 2;

        // Second conv2D (size W/2 x H/2)
        gpu_conv2D(b, _encoder_filter_2.get(), a,
                   n_batch, w_curr, h_curr, _ENCODER_FILTER_1_DEPTH, _ENCODER_FILTER_2_DEPTH);

        gpu_add_bias(a, _encoder_bias_2.get(), b,
                     n_batch, w_curr, h_curr, _ENCODER_FILTER_2_DEPTH);

        // Second ReLU
        gpu_relu(b, a, n_batch, w_curr, h_curr, _ENCODER_FILTER_2_DEPTH);

        // Second max pooling (size W/4 x H/4)
        gpu_max_pooling(a, b, n_batch, w_curr, h_curr, _ENCODER_FILTER_2_DEPTH);
        // Final result in b (w_curr/h_curr is not needed further here)

        // Copy batch
        int n_floats_batch = n_batch * encoded_image_elements;
        CUDA_CHECK(cudaMemcpy(res.get_data() + offset, b, n_floats_batch * sizeof(float), cudaMemcpyDeviceToHost));
        offset += n_floats_batch; // Tăng offset bằng số phần tử (floats)
    }

    // Copy labels
    CUDA_CHECK(cudaMemcpy(res.get_labels(), dataset.get_labels(), dataset.n * sizeof(int), cudaMemcpyHostToHost));

    cudaFree(a);
    cudaFree(b);

    return res;
}
Dataset Gpu_Autoencoder::decode(const Dataset &dataset) const {
    int width = dataset.width, height = dataset.height, depth = dataset.depth; // width, height là W/4, H/4
    
    // Decoded output is W x H (original size). Input latent is W/4 x H/4.
    int W_orig = IMAGE_WIDTH; // 32
    int H_orig = IMAGE_HEIGHT; // 32

    // Total elements in the decoded image (W * H * D3)
    int decoded_image_elements = W_orig * H_orig * _DECODER_FILTER_3_DEPTH;
    int offset = 0;

    vector<Dataset> batches = create_minibatches(dataset, _ENCODE_BATCH_SIZE);
    Dataset res(dataset.n, W_orig, H_orig, _DECODER_FILTER_3_DEPTH);

    // Placeholder, alternating - Allocate memory for W x H output
    float *a, *b;
    int n_pixel_max = _ENCODE_BATCH_SIZE * W_orig * H_orig;
    int max_depth_intermediate = std::max(_DECODER_FILTER_3_DEPTH, (int)MAX_FILTER_DEPTH);
    CUDA_CHECK(cudaMalloc(&a, n_pixel_max * max_depth_intermediate * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, n_pixel_max * max_depth_intermediate * sizeof(float)));

    for (size_t i = 0; i < batches.size(); ++i) {
        int n_batch = batches[i].n;
        int w_curr = width;  // W/4
        int h_curr = height; // H/4
        
        // First conv2D (Output: W/4 x H/4 x D1)
        gpu_conv2D(batches[i].get_data(), _decoder_filter_1.get(), a,
                    n_batch, w_curr, h_curr, depth, _DECODER_FILTER_1_DEPTH);

        // Add bias
        gpu_add_bias(a, _decoder_bias_1.get(), b,
                    n_batch, w_curr, h_curr, _DECODER_FILTER_1_DEPTH);

        // ReLU
        gpu_relu(b, a, n_batch, w_curr, h_curr, _DECODER_FILTER_1_DEPTH);

        // Upsampling 1 (Output: W/2 x H/2 x D1)
        w_curr *= 2; // W/2
        h_curr *= 2; // H/2
        gpu_upsampling(a, b, n_batch, w_curr, h_curr, _DECODER_FILTER_1_DEPTH);

        // Second conv2D (Output: W/2 x H/2 x D2)
        gpu_conv2D(b, _decoder_filter_2.get(), a,
                    n_batch, w_curr, h_curr, _DECODER_FILTER_1_DEPTH, _DECODER_FILTER_2_DEPTH);

        gpu_add_bias(a, _decoder_bias_2.get(), b,
                    n_batch, w_curr, h_curr, _DECODER_FILTER_2_DEPTH);

        // Second ReLU
        gpu_relu(b, a, n_batch, w_curr, h_curr, _DECODER_FILTER_2_DEPTH);

        // Second upsampling (Output: W x H x D2)
        w_curr *= 2; // W
        h_curr *= 2; // H
        gpu_upsampling(a, b, n_batch, w_curr, h_curr, _DECODER_FILTER_2_DEPTH);

        // Third conv2D (Final Output: W x H x D3)
        gpu_conv2D(b, _decoder_filter_3.get(), a,
                    n_batch, w_curr, h_curr, _DECODER_FILTER_2_DEPTH, _DECODER_FILTER_3_DEPTH);

        gpu_add_bias(a, _decoder_bias_3.get(), b,
                    n_batch, w_curr, h_curr, _DECODER_FILTER_3_DEPTH);

        // Copy batch
        int n_floats_batch = n_batch * decoded_image_elements;
        CUDA_CHECK(cudaMemcpy(res.get_data() + offset, b, n_floats_batch * sizeof(float), cudaMemcpyDeviceToHost));
        offset += n_floats_batch;
    }

    // Copy the result
    CUDA_CHECK(cudaMemcpy(res.get_labels(), dataset.get_labels(), dataset.n * sizeof(int), cudaMemcpyHostToHost));
    
    cudaFree(a);
    cudaFree(b);
    
    return res;
}

float Gpu_Autoencoder::eval(const Dataset &dataset) {
    int depth_out = _DECODER_FILTER_3_DEPTH;
    Dataset decoded = decode(encode(dataset));
    return gpu_mse_loss(dataset.get_data(), decoded.get_data(),
                        dataset.n, dataset.width, dataset.height, depth_out); 
}