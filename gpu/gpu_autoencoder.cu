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
            printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        }                                                        \
    }

Gpu_Autoencoder::Gpu_Autoencoder(): IAutoencoder() {};
Gpu_Autoencoder::Gpu_Autoencoder(const char *filename): IAutoencoder(filename) {};

// -----------------------------------------------------
// FORWARD PASS HELPERS
// -----------------------------------------------------

Dataset Gpu_Autoencoder::_encode_save_output(const Dataset &dataset)
{
    int n = dataset.n, width = dataset.width, height = dataset.height, depth = dataset.depth;

    // First conv2D layer
    gpu_conv2D(dataset.get_data(), _encoder_filter_1.get(), _out_encoder_filter_1.get(),
               n, width, height, depth, _ENCODER_FILTER_1_DEPTH);

    // Dim: n * w * h * F1
    gpu_add_bias(_out_encoder_filter_1.get(), _encoder_bias_1.get(), _out_encoder_bias_1.get(),
                 n, width, height, _ENCODER_FILTER_1_DEPTH);

    // ReLU layer
    gpu_relu(_out_encoder_bias_1.get(), _out_encoder_relu_1.get(),
             n, width, height, _ENCODER_FILTER_1_DEPTH);

    // First max pooling layer
    gpu_max_pooling(_out_encoder_relu_1.get(), _out_max_pooling_1.get(),
                    n, width, height, _ENCODER_FILTER_1_DEPTH);

    // Dim: n * w/2 * h/2 * F1
    // Second conv2D layer
    gpu_conv2D(_out_max_pooling_1.get(), _encoder_filter_2.get(), _out_encoder_filter_2.get(),
               n, width / 2, height / 2, _ENCODER_FILTER_1_DEPTH, _ENCODER_FILTER_2_DEPTH);

    // Dim: n * w/2 * h/2 * F2
    gpu_add_bias(_out_encoder_filter_2.get(), _encoder_bias_2.get(), _out_encoder_bias_2.get(),
                 n, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

    // ReLU layer
    gpu_relu(_out_encoder_bias_2.get(), _out_encoder_relu_2.get(),
             n, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

    // Second max pooling layer
    gpu_max_pooling(_out_encoder_relu_2.get(), _out_max_pooling_2.get(),
                    n, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

    // Return the result (Dim: n * w/4 * h/4 * F2)
    int n_elements = n * width / 4 * height / 4 * _ENCODER_FILTER_2_DEPTH;
    Dataset res(n, width / 4, height / 4, _ENCODER_FILTER_2_DEPTH);
    cudaError_t err = cudaMemcpy(res.get_data(), _out_max_pooling_2.get(),
                                 n_elements * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    return res;
}

Dataset Gpu_Autoencoder::_decode_save_output(const Dataset &dataset)
{
    int n = dataset.n, width = dataset.width, height = dataset.height, depth = dataset.depth;
    
    // Input is Latent Space: n * w/4 * h/4 * F2 (width, height = W/4, H/4)

    // First conv2D layer (Output Dim: n * w/4 * h/4 * D1)
    gpu_conv2D(dataset.get_data(), _decoder_filter_1.get(), _out_decoder_filter_1.get(),
               n, width, height, depth, _DECODER_FILTER_1_DEPTH);

    // Dim: n * w * h * D1
    gpu_add_bias(_out_decoder_filter_1.get(), _decoder_bias_1.get(), _out_decoder_bias_1.get(),
                 n, width, height, _DECODER_FILTER_1_DEPTH);

    // ReLU layer
    gpu_relu(_out_decoder_bias_1.get(), _out_decoder_relu_1.get(),
             n, width, height, _DECODER_FILTER_1_DEPTH);

    // First upsampling layer (Output Dim: n * 2w * 2h * D1)
    // Note: If input width/height is W/4, output is W/2 * H/2
    int w_half = width * 2; // W/2
    int h_half = height * 2; // H/2
    gpu_upsampling(_out_decoder_relu_1.get(), _out_upsampling_1.get(),
                   n, w_half, h_half, _DECODER_FILTER_1_DEPTH);

    // Second conv2D layer (Output Dim: n * 2w * 2h * D2)
    gpu_conv2D(_out_upsampling_1.get(), _decoder_filter_2.get(), _out_decoder_filter_2.get(),
               n, w_half, h_half, _DECODER_FILTER_1_DEPTH, _DECODER_FILTER_2_DEPTH);

    // Dim: n * 2w * 2h * D2
    gpu_add_bias(_out_decoder_filter_2.get(), _decoder_bias_2.get(), _out_decoder_bias_2.get(),
                 n, w_half, h_half, _DECODER_FILTER_2_DEPTH);

    // ReLU layer
    gpu_relu(_out_decoder_bias_2.get(), _out_decoder_relu_2.get(),
             n, w_half, h_half, _DECODER_FILTER_2_DEPTH);

    // Second upsampling layer (Output Dim: n * 4w * 4h * D2)
    // Note: If input width/height is W/2, output is W * H
    int w_full = w_half * 2; // W
    int h_full = h_half * 2; // H
    gpu_upsampling(_out_decoder_relu_2.get(), _out_upsampling_2.get(),
                   n, w_full, h_full, _DECODER_FILTER_2_DEPTH);

    // Third conv2D layer (Final Output Dim: n * 4w * 4h * D3)
    // Note: This output is W * H * D3
    gpu_conv2D(_out_upsampling_2.get(), _decoder_filter_3.get(), _out_decoder_filter_3.get(),
               n, w_full, h_full, _DECODER_FILTER_2_DEPTH, _DECODER_FILTER_3_DEPTH);

    // Dim: n * 4w * 4h * D3
    gpu_add_bias(_out_decoder_filter_3.get(), _decoder_bias_3.get(), _out_decoder_bias_3.get(),
                 n, w_full, h_full, _DECODER_FILTER_3_DEPTH);

    // Return the result (Dim: n * W * H * D3)
    Dataset res(n, w_full, h_full, _DECODER_FILTER_3_DEPTH);

    // FIX Bug 2: Corrected size calculation. Total elements = n * (4*width) * (4*height) * depth
    // The sizes w_full and h_full are now W and H (original image dimensions).
    int total_output_elements = n * w_full * h_full * _DECODER_FILTER_3_DEPTH;

    cudaError_t err = cudaMemcpy(res.get_data(), _out_decoder_bias_3.get(),
                                 total_output_elements * sizeof(float),
                                 cudaMemcpyDeviceToHost);
    CUDA_CHECK(err);

    return res;
}

// -----------------------------------------------------
// MEMORY MANAGEMENT
// -----------------------------------------------------

void Gpu_Autoencoder::_allocate_output_mem(int n, int width, int height)
{
    // Calculate different pixel counts for different resolution levels
    int n_pixel = n * width * height;                     // Original: n * w * h (32x32)
    int n_pixel_half = n * (width / 2) * (height / 2);    // After 1st pool: n * w/2 * h/2 (16x16)
    int n_pixel_quarter = n * (width / 4) * (height / 4); // After 2nd pool: n * w/4 * h/4 (8x8)

    // Max dimension for temporary buffers (d_in, d_out) is the output of the decoder (n * w * h * D3)
    // Since the fixed logic outputs W*H, the max pixel count is n_pixel.
    int n_pixel_max_tensor = n_pixel; 
    int max_depth_intermediate = std::max({_DECODER_FILTER_3_DEPTH, _DECODER_FILTER_2_DEPTH, _ENCODER_FILTER_1_DEPTH});

    float *d_ptr = nullptr;

    // --- Encoder output allocations (UNCHANGED/ASSUMED CORRECT) ---
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

    // --- Decoder output allocations (UNCHANGED/ASSUMED CORRECT) ---
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

    // --- Allocate temporary buffers (FIX Bug 4: Critical Memory Allocation Issue) ---
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
// TRAINING AND EVALUATION
// -----------------------------------------------------

float Gpu_Autoencoder::_fit_batch(const Dataset &batch, float learning_rate)
{
    // Get the result after autoencoding
    int n = batch.n, width = batch.width, height = batch.height; // width, height = 32, 32
    // Use depth of the decoded output for comparison (depth_out = 3)
    int depth_out = _DECODER_FILTER_3_DEPTH;

    float *d_in = _d_in.get(), *d_out = _d_out.get(), *d_filter = _d_filter.get();
    Dataset encoded = _encode_save_output(batch); // Output: n * 8 * 8 * F2
    Dataset res = _decode_save_output(encoded);   // Output: n * 32 * 32 * D3

    // Calculate loss before backprop
    // FIX Bug 1: Use correct output dimensions (W x H) for comparison with input batch.
    float loss = gpu_mse_loss(batch.get_data(), res.get_data(),
                              n, width, height, depth_out); // Use W, H of the input batch (32x32)

    // Get loss gradient
    // FIX Bug 1: Use correct output dimensions (W x H)
    gpu_mse_grad(batch.get_data(), res.get_data(), d_out,
                 n, width, height, depth_out); // Use W, H of the input batch (32x32)

    // Update weight for the last conv2D layer (size W x H x D3)
    // Update bias
    gpu_bias_grad(d_out, d_in, n, width, height, depth_out);

    gpu_update_weight(_decoder_bias_3.get(), d_in, depth_out, learning_rate);

    // Update filter (Input to layer: n * W * H * D2)
    // d_out size: n * W * H * D3
    // W, H are 32, 32
    gpu_conv2D_grad(_out_upsampling_2.get(), d_out, d_filter,
                    n, width, height, _DECODER_FILTER_2_DEPTH, depth_out);

    // Pass delta backwards (gpu_conv2D is convolution with swapped inputs/outputs for backprop)
    // Output d_in: n * W * H * D2
    gpu_conv2D(d_out, _decoder_filter_3.get(), d_in,
               n, width, height, _DECODER_FILTER_2_DEPTH, depth_out);

    // Swap d_out and d_in
    swap(d_out, d_in);

    // Update weight
    gpu_update_weight(_decoder_filter_3.get(), d_filter, _DECODER_FILTER_3_SIZE, learning_rate);

    // Pass through upsampling (dim: n * W/2 * H/2 * D2)
    // d_out size: n * W * H * D2
    int w_half = width / 2; // 16
    int h_half = height / 2; // 16
    gpu_upsampling_backward(d_out, d_in, n, w_half * 2, h_half * 2, _DECODER_FILTER_2_DEPTH);

    // Pass through ReLU (d_in and d_out swapped) (size W/2 x H/2 x D2)
    gpu_relu_backward(_out_decoder_bias_2.get(), d_in, d_out,
                      n, w_half, h_half, _DECODER_FILTER_2_DEPTH);

    // Second conv2D layer (size W/2 x H/2 x D2)
    gpu_bias_grad(d_out, d_in, n, w_half, h_half, _DECODER_FILTER_2_DEPTH);

    gpu_update_weight(_decoder_bias_2.get(), d_in, _DECODER_FILTER_2_DEPTH, learning_rate);

    // Update filter (Input to layer: n * W/2 * H/2 * D1)
    // d_out size: n * W/2 * H/2 * D2
    gpu_conv2D_grad(_out_upsampling_1.get(), d_out, d_filter,
                    n, w_half, h_half, _DECODER_FILTER_1_DEPTH, _DECODER_FILTER_2_DEPTH);

    // Output d_in: n * W/2 * H/2 * D1
    gpu_conv2D(d_out, _decoder_filter_2.get(), d_in,
               n, w_half, h_half, _DECODER_FILTER_1_DEPTH, _DECODER_FILTER_2_DEPTH);

    swap(d_out, d_in);
    gpu_update_weight(_decoder_filter_2.get(), d_filter, _DECODER_FILTER_2_SIZE, learning_rate);

    // Upsampling (dim: n * W/4 * H/4 * D1)
    // d_out size: n * W/2 * H/2 * D1
    int w_quarter = width / 4; // 8
    int h_quarter = height / 4; // 8
    gpu_upsampling_backward(d_out, d_in, n, w_quarter * 2, h_quarter * 2, _DECODER_FILTER_1_DEPTH);

    // ReLU (size W/4 x H/4 x D1)
    gpu_relu_backward(_out_decoder_bias_1.get(), d_in, d_out,
                      n, w_quarter, h_quarter, _DECODER_FILTER_1_DEPTH);

    // Third Conv2D (first decoder layer) (size W/4 x H/4 x D1)
    gpu_bias_grad(d_out, d_in, n, w_quarter, h_quarter, _DECODER_FILTER_1_DEPTH);

    gpu_update_weight(_decoder_bias_1.get(), d_in, _DECODER_FILTER_1_DEPTH, learning_rate);

    // Update filter (Input to layer: n * W/4 * H/4 * F2)
    // d_out size: n * W/4 * H/4 * D1
    gpu_conv2D_grad(_out_max_pooling_2.get(), d_out, d_filter,
                    n, w_quarter, h_quarter, _ENCODER_FILTER_2_DEPTH, _DECODER_FILTER_1_DEPTH);

    // Output d_in: n * W/4 * H/4 * F2 (Latent space gradient)
    gpu_conv2D(d_out, _decoder_filter_1.get(), d_in,
               n, w_quarter, h_quarter, _ENCODER_FILTER_2_DEPTH, _DECODER_FILTER_1_DEPTH);

    swap(d_out, d_in);
    gpu_update_weight(_decoder_filter_1.get(), d_filter, _DECODER_FILTER_1_SIZE, learning_rate);

    // --------------------------------------------------------------------------------
    // ENCODER BACKWARD PASS
    // --------------------------------------------------------------------------------
    
    // Max pooling backwards (dim: n * W/2 * H/2 * F2)
    // d_out size: n * W/4 * H/4 * F2
    gpu_max_pooling_backward(_out_encoder_relu_2.get(), d_out, d_in,
                             n, w_half, h_half, _ENCODER_FILTER_2_DEPTH);

    // ReLU (size W/2 x H/2 x F2)
    gpu_relu_backward(_out_encoder_bias_2.get(), d_in, d_out,
                      n, w_half, h_half, _ENCODER_FILTER_2_DEPTH);

    // Fourth conv2D (second encoder layer) (size W/2 x H/2 x F2)
    gpu_bias_grad(d_out, d_in, n, w_half, h_half, _ENCODER_FILTER_2_DEPTH);

    gpu_update_weight(_encoder_bias_2.get(), d_in, _ENCODER_FILTER_2_DEPTH, learning_rate);

    // Update filter (Input to layer: n * W/2 * H/2 * F1)
    // d_out size: n * W/2 * H/2 * F2
    gpu_conv2D_grad(_out_max_pooling_1.get(), d_out, d_filter,
                    n, w_half, h_half, _ENCODER_FILTER_1_DEPTH, _ENCODER_FILTER_2_DEPTH);

    // Output d_in: n * W/2 * H/2 * F1
    gpu_conv2D(d_out, _encoder_filter_2.get(), d_in,
               n, w_half, h_half, _ENCODER_FILTER_1_DEPTH, _ENCODER_FILTER_2_DEPTH);

    swap(d_out, d_in);
    gpu_update_weight(_encoder_filter_2.get(), d_filter, _ENCODER_FILTER_2_SIZE, learning_rate);

    // First max pooling backwards (dim: n * W * H * F1)
    // d_out size: n * W/2 * H/2 * F1
    gpu_max_pooling_backward(_out_encoder_relu_1.get(), d_out, d_in,
                             n, width, height, _ENCODER_FILTER_1_DEPTH);

    // ReLU (size W x H x F1)
    gpu_relu_backward(_out_encoder_bias_1.get(), d_in, d_out,
                      n, width, height, _ENCODER_FILTER_1_DEPTH);

    // Fifth conv2D (first encoder layer) (size W x H x F1)
    gpu_bias_grad(d_out, d_in, n, width, height, _ENCODER_FILTER_1_DEPTH);

    gpu_update_weight(_encoder_bias_1.get(), d_in, _ENCODER_FILTER_1_DEPTH, learning_rate);

    // Update filter (Input to layer: n * W * H * D_input)
    // d_out size: n * W * H * F1
    gpu_conv2D_grad(batch.get_data(), d_out, d_filter,
                    n, width, height, batch.depth, _ENCODER_FILTER_1_DEPTH);

    // Final output d_in: n * W * H * D_input (gradient to input, typically discarded)
    gpu_conv2D(d_out, _encoder_filter_1.get(), d_in,
               n, width, height, batch.depth, _ENCODER_FILTER_1_DEPTH);

    swap(d_out, d_in);
    gpu_update_weight(_encoder_filter_1.get(), d_filter, _ENCODER_FILTER_1_SIZE, learning_rate);

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
// ENCODE/DECODE (Batch Processing)
// -----------------------------------------------------

Dataset Gpu_Autoencoder::encode(const Dataset &dataset) const
{
    // Encode by batches to use less memory
    int width = dataset.width, height = dataset.height, depth = dataset.depth;
    // Total elements in the encoded image
    int encoded_image_elements = width / 4 * height / 4 * _ENCODER_FILTER_2_DEPTH;
    int offset = 0; // Offset tính bằng số float (phần tử)

    vector<Dataset> batches = create_minibatches(dataset, _ENCODE_BATCH_SIZE);
    Dataset res(dataset.n, width / 4, height / 4, _ENCODER_FILTER_2_DEPTH);

    // Placeholder, alternating
    float *a, *b;
    // Max image size for intermediates at W/2 x H/2 resolution * MAX_FILTER_DEPTH
    int n_pixel_max = _ENCODE_BATCH_SIZE * width * height / 4; 
    CUDA_CHECK(cudaMalloc(&a, n_pixel_max * MAX_FILTER_DEPTH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, n_pixel_max * MAX_FILTER_DEPTH * sizeof(float)));

    for (size_t i = 0; i < batches.size(); ++i)
    {
        int n_batch = batches[i].n;

        // First conv2D (size W x H)
        gpu_conv2D(batches[i].get_data(), _encoder_filter_1.get(), a,
                   n_batch, width, height, depth, _ENCODER_FILTER_1_DEPTH);

        // Add bias
        gpu_add_bias(a, _encoder_bias_1.get(), b,
                     n_batch, width, height, _ENCODER_FILTER_1_DEPTH);

        // ReLU
        gpu_relu(b, a, n_batch, width, height, _ENCODER_FILTER_1_DEPTH);

        // Max pooling (size W/2 x H/2)
        gpu_max_pooling(a, b, n_batch, width, height, _ENCODER_FILTER_1_DEPTH);

        // Second conv2D (size W/2 x H/2)
        gpu_conv2D(b, _encoder_filter_2.get(), a,
                   n_batch, width / 2, height / 2, _ENCODER_FILTER_1_DEPTH, _ENCODER_FILTER_2_DEPTH);

        gpu_add_bias(a, _encoder_bias_2.get(), b,
                     n_batch, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

        // Second ReLU
        gpu_relu(b, a, n_batch, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

        // Second max pooling (size W/4 x H/4)
        gpu_max_pooling(a, b, n_batch, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

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
    int width = dataset.width, height = dataset.height, depth = dataset.depth;
    
    // Decoded output is W x H (original size). Input latent is W/4 x H/4.
    int W_orig = IMAGE_WIDTH; // 32
    int H_orig = IMAGE_HEIGHT; // 32

    // Total elements in the decoded image (W * H * D3)
    int decoded_image_elements = W_orig * H_orig * _DECODER_FILTER_3_DEPTH;
    int offset  = 0;

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

        // Input is width * height (W/4 x H/4)
        int w_q = width;
        int h_q = height;
        
        // First conv2D (Output: W/4 x H/4 x D1)
        gpu_conv2D(batches[i].get_data(), _decoder_filter_1.get(), a,
                    n_batch, w_q, h_q, depth, _DECODER_FILTER_1_DEPTH);

        // Add bias
        gpu_add_bias(a, _decoder_bias_1.get(), b,
                    n_batch, w_q, h_q, _DECODER_FILTER_1_DEPTH);

        // ReLU
        gpu_relu(b, a, n_batch, w_q, h_q, _DECODER_FILTER_1_DEPTH);

        // Upsampling (Output: W/2 x H/2 x D1)
        int w_h = w_q * 2;
        int h_h = h_q * 2;
        gpu_upsampling(a, b, n_batch, w_h, h_h, _DECODER_FILTER_1_DEPTH);

        // Second conv2D (Output: W/2 x H/2 x D2)
        gpu_conv2D(b, _decoder_filter_2.get(), a,
                    n_batch, w_h, h_h, _DECODER_FILTER_1_DEPTH, _DECODER_FILTER_2_DEPTH);

        gpu_add_bias(a, _decoder_bias_2.get(), b,
                    n_batch, w_h, h_h, _DECODER_FILTER_2_DEPTH);

        // Second ReLU
        gpu_relu(b, a, n_batch, w_h, h_h, _DECODER_FILTER_2_DEPTH);

        // Second upsampling (Output: W x H x D2)
        int w_f = w_h * 2;
        int h_f = h_h * 2;
        gpu_upsampling(a, b, n_batch, w_f, h_f, _DECODER_FILTER_2_DEPTH);

        // Third conv2D (Final Output: W x H x D3)
        gpu_conv2D(b, _decoder_filter_3.get(), a,
                    n_batch, w_f, h_f, _DECODER_FILTER_2_DEPTH, _DECODER_FILTER_3_DEPTH);

        gpu_add_bias(a, _decoder_bias_3.get(), b,
                    n_batch, w_f, h_f, _DECODER_FILTER_3_DEPTH);

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
    return gpu_mse_loss(dataset.get_data(), decode(encode(dataset)).get_data(),
                        dataset.n, dataset.width, dataset.height, depth_out); // Use W, H of the input batch (32x32)
}