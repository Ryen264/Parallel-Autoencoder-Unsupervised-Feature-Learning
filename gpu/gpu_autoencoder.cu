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

// Assuming these are implemented in IAutoencoder or are provided elsewhere
// Gpu_Autoencoder::Gpu_Autoencoder(): IAutoencoder() {};
// Gpu_Autoencoder::Gpu_Autoencoder(const char *filename): IAutoencoder(filename) {};

// -----------------------------------------------------
// FORWARD PASS HELPERS
// -----------------------------------------------------

Dataset Gpu_Autoencoder::_encode_save_output(const Dataset &dataset)
{
    int n = dataset.n, width = dataset.width, height = dataset.height, depth = dataset.depth;

    // First conv2D layer
    gpu_conv2D(dataset.get_data(), _encoder_filter_1.get(), _out_encoder_filter_1.get(),
               n, width, height, depth, _ENCODER_FILTER_1_DEPTH);

    // Dim: n * w * w * F1
    gpu_add_bias(_out_encoder_filter_1.get(), _encoder_bias_1.get(), _out_encoder_bias_1.get(),
                 n, width, height, _ENCODER_FILTER_1_DEPTH);

    // ReLU layer
    gpu_relu(_out_encoder_bias_1.get(), _out_encoder_relu_1.get(),
             n, width, height, _ENCODER_FILTER_1_DEPTH);

    // First max pooling layer
    gpu_max_pooling(_out_encoder_relu_1.get(), _out_max_pooling_1.get(),
                    n, width, height, _ENCODER_FILTER_1_DEPTH);

    // Dim: n * w/2 * w/2 * F1
    // Second conv2D layer
    gpu_conv2D(_out_max_pooling_1.get(), _encoder_filter_2.get(), _out_encoder_filter_2.get(),
               n, width / 2, height / 2, _ENCODER_FILTER_1_DEPTH, _ENCODER_FILTER_2_DEPTH);

    // Dim: n * w/2 * w/2 * F2
    gpu_add_bias(_out_encoder_filter_2.get(), _encoder_bias_2.get(), _out_encoder_bias_2.get(),
                 n, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

    // ReLU layer
    gpu_relu(_out_encoder_bias_2.get(), _out_encoder_relu_2.get(),
             n, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

    // Second max pooling layer
    gpu_max_pooling(_out_encoder_relu_2.get(), _out_max_pooling_2.get(),
                    n, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

    // Return the result (Dim: n * w/4 * w/4 * F2)
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

    // First conv2D layer
    gpu_conv2D(dataset.get_data(), _decoder_filter_1.get(), _out_decoder_filter_1.get(),
               n, width, height, depth, _DECODER_FILTER_1_DEPTH);

    // Dim: n * w * w * D1
    gpu_add_bias(_out_decoder_filter_1.get(), _decoder_bias_1.get(), _out_decoder_bias_1.get(),
                 n, width, height, _DECODER_FILTER_1_DEPTH);

    // ReLU layer
    gpu_relu(_out_decoder_bias_1.get(), _out_decoder_relu_1.get(),
             n, width, height, _DECODER_FILTER_1_DEPTH);

    // First upsampling layer (Output Dim: n * 2w * 2h * D1)
    gpu_upsampling(_out_decoder_relu_1.get(), _out_upsampling_1.get(),
                   n, width, height, _DECODER_FILTER_1_DEPTH);

    // Second conv2D layer
    gpu_conv2D(_out_upsampling_1.get(), _decoder_filter_2.get(), _out_decoder_filter_2.get(),
               n, 2 * width, 2 * height, _DECODER_FILTER_1_DEPTH, _DECODER_FILTER_2_DEPTH);

    // Dim: n * 2w * 2w * D2
    gpu_add_bias(_out_decoder_filter_2.get(), _decoder_bias_2.get(), _out_decoder_bias_2.get(),
                 n, 2 * width, 2 * height, _DECODER_FILTER_2_DEPTH);

    // ReLU layer
    gpu_relu(_out_decoder_bias_2.get(), _out_decoder_relu_2.get(),
             n, 2 * width, 2 * height, _DECODER_FILTER_2_DEPTH);

    // Second upsampling layer (Output Dim: n * 4w * 4h * D2)
    gpu_upsampling(_out_decoder_relu_2.get(), _out_upsampling_2.get(),
                   n, 2 * width, 2 * height, _DECODER_FILTER_2_DEPTH);

    // Third conv2D layer
    gpu_conv2D(_out_upsampling_2.get(), _decoder_filter_3.get(), _out_decoder_filter_3.get(),
               n, 4 * width, 4 * height, _DECODER_FILTER_2_DEPTH, _DECODER_FILTER_3_DEPTH);

    // Dim: n * 4w * 4w * D3
    gpu_add_bias(_out_decoder_filter_3.get(), _decoder_bias_3.get(), _out_decoder_bias_3.get(),
                 n, 4 * width, 4 * height, _DECODER_FILTER_3_DEPTH);

    // Return the result (Dim: n * 4w * 4w * D3)
    Dataset res(n, 4 * width, 4 * height, _DECODER_FILTER_3_DEPTH);

    // FIX Bug 2: Corrected size calculation. Total elements = n * (4*width) * (4*height) * depth
    int total_output_elements = n * (4 * width) * (4 * height) * _DECODER_FILTER_3_DEPTH;

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
    int n_pixel = n * width * height;                     // Original: n * w * h
    int n_pixel_half = n * (width / 2) * (height / 2);    // After 1st pool: n * w/2 * h/2
    int n_pixel_quarter = n * (width / 4) * (height / 4); // After 2nd pool: n * w/4 * h/4

    // Max dimension for temporary buffers (d_in, d_out) is the output of the decoder (n * 4w * 4h * D3)
    int n_pixel_4x = n * (4 * width) * (4 * height);
    int max_depth_intermediate = std::max(_DECODER_FILTER_3_DEPTH, (int)MAX_FILTER_DEPTH);

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

    // _d_in and _d_out must hold the largest tensor size (4x upsampled output)
    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_4x * max_depth_intermediate * sizeof(float)));
    _d_in = std::shared_ptr<float>(d_ptr, [](float *p)
                                   { cudaFree(p); });

    CUDA_CHECK(cudaMalloc(&d_ptr, n_pixel_4x * max_depth_intermediate * sizeof(float)));
    _d_out = std::shared_ptr<float>(d_ptr, [](float *p)
                                    { cudaFree(p); });

    CUDA_CHECK(cudaMalloc(&d_ptr, MAX_FILTER_SIZE * sizeof(float)));
    _d_filter = std::shared_ptr<float>(d_ptr, [](float *p)
                                       { cudaFree(p); });
}

void Gpu_Autoencoder::_deallocate_output_mem()
{
    // ... (UNCHANGED/ASSUMED CORRECT)
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
    int n = batch.n, width = batch.width, height = batch.height;
    // Use depth of the decoded output for comparison
    int depth_out = _DECODER_FILTER_3_DEPTH;

    float *d_in = _d_in.get(), *d_out = _d_out.get(), *d_filter = _d_filter.get();
    Dataset res = _decode_save_output(_encode_save_output(batch));

    // Calculate loss before backprop
    // FIX Bug 1: Use correct output depth (depth_out) and dimensions
    float loss = gpu_mse_loss(batch.get_data(), res.get_data(),
                              n, 4 * width, 4 * height, depth_out);

    // Get loss gradient
    // FIX Bug 1: Use correct output depth (depth_out) and dimensions
    gpu_mse_grad(batch.get_data(), res.get_data(), d_out,
                 n, 4 * width, 4 * height, depth_out);

    // Update weight for the last conv2D layer
    // Update bias
    gpu_bias_grad(d_out, d_in, n, 4 * width, 4 * height, depth_out);

    gpu_update_weight(_decoder_bias_3.get(), d_in, depth_out, learning_rate);

    // Update filter
    gpu_conv2D_grad(_out_upsampling_2.get(), d_out, d_filter,
                    n, 4 * width, 4 * height, _DECODER_FILTER_2_DEPTH, depth_out);

    // Pass delta backwards (gpu_conv2D is convolution with swapped inputs/outputs for backprop)
    gpu_conv2D(d_out, _decoder_filter_3.get(), d_in,
               n, 4 * width, 4 * height, _DECODER_FILTER_2_DEPTH, depth_out);

    // Swap d_out and d_in
    swap(d_out, d_in);

    // Update weight
    gpu_update_weight(_decoder_filter_3.get(), d_filter, _DECODER_FILTER_3_SIZE, learning_rate);

    // Pass through upsampling (dim: n * 2w * 2w * D2)
    gpu_upsampling_backward(d_out, d_in, n, 2 * width, 2 * height, _DECODER_FILTER_2_DEPTH);

    // Pass through ReLU (d_in and d_out swapped)
    gpu_relu_backward(_out_decoder_bias_2.get(), d_in, d_out,
                      n, 2 * width, 2 * height, _DECODER_FILTER_2_DEPTH);

    // Second conv2D layer
    gpu_bias_grad(d_out, d_in, n, 2 * width, 2 * height, _DECODER_FILTER_2_DEPTH);

    gpu_update_weight(_decoder_bias_2.get(), d_in, _DECODER_FILTER_2_DEPTH, learning_rate);

    gpu_conv2D_grad(_out_upsampling_1.get(), d_out, d_filter,
                    n, 2 * width, 2 * height, _DECODER_FILTER_1_DEPTH, _DECODER_FILTER_2_DEPTH);

    gpu_conv2D(d_out, _decoder_filter_2.get(), d_in,
               n, 2 * width, 2 * height, _DECODER_FILTER_1_DEPTH, _DECODER_FILTER_2_DEPTH);

    swap(d_out, d_in);
    gpu_update_weight(_decoder_filter_2.get(), d_filter, _DECODER_FILTER_2_SIZE, learning_rate);

    // Upsampling (dim: n * w * w * D1)
    gpu_upsampling_backward(d_out, d_in, n, width, height, _DECODER_FILTER_1_DEPTH);

    // ReLU
    gpu_relu_backward(_out_decoder_bias_1.get(), d_in, d_out,
                      n, width, height, _DECODER_FILTER_1_DEPTH);

    // Third Conv2D
    gpu_bias_grad(d_out, d_in, n, width, height, _DECODER_FILTER_1_DEPTH);

    gpu_update_weight(_decoder_bias_1.get(), d_in, _DECODER_FILTER_1_DEPTH, learning_rate);

    gpu_conv2D_grad(_out_max_pooling_2.get(), d_out, d_filter,
                    n, width, height, _ENCODER_FILTER_2_DEPTH, _DECODER_FILTER_1_DEPTH);

    gpu_conv2D(d_out, _decoder_filter_1.get(), d_in,
               n, width, height, _ENCODER_FILTER_2_DEPTH, _DECODER_FILTER_1_DEPTH);

    swap(d_out, d_in);
    gpu_update_weight(_decoder_filter_1.get(), d_filter, _DECODER_FILTER_1_SIZE, learning_rate);

    // Max pooling backwards (dim: n * w/2 * w/2 * F2)
    gpu_max_pooling_backward(_out_encoder_relu_2.get(), d_out, d_in,
                             n, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

    gpu_relu_backward(_out_encoder_bias_2.get(), d_in, d_out,
                      n, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

    // Fourth conv2D
    gpu_bias_grad(d_out, d_in, n, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

    gpu_update_weight(_encoder_bias_2.get(), d_in, _ENCODER_FILTER_2_DEPTH, learning_rate);

    gpu_conv2D_grad(_out_max_pooling_1.get(), d_out, d_filter,
                    n, width / 2, height / 2, _ENCODER_FILTER_1_DEPTH, _ENCODER_FILTER_2_DEPTH);

    gpu_conv2D(d_out, _encoder_filter_2.get(), d_in,
               n, width / 2, height / 2, _ENCODER_FILTER_1_DEPTH, _ENCODER_FILTER_2_DEPTH);

    swap(d_out, d_in);
    gpu_update_weight(_encoder_filter_2.get(), d_filter, _ENCODER_FILTER_2_SIZE, learning_rate);

    // First max pooling backwards (dim: n * w * w * F1)
    gpu_max_pooling_backward(_out_encoder_relu_1.get(), d_out, d_in,
                             n, width, height, _ENCODER_FILTER_1_DEPTH);

    gpu_relu_backward(_out_encoder_bias_1.get(), d_in, d_out,
                      n, width, height, _ENCODER_FILTER_1_DEPTH);

    // Fifth conv2D (first encoder layer)
    gpu_bias_grad(d_out, d_in, n, width, height, _ENCODER_FILTER_1_DEPTH);

    gpu_update_weight(_encoder_bias_1.get(), d_in, _ENCODER_FILTER_1_DEPTH, learning_rate);

    gpu_conv2D_grad(batch.get_data(), d_out, d_filter,
                    n, width, height, batch.depth, _ENCODER_FILTER_1_DEPTH);

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
    int n_pixel_max = _ENCODE_BATCH_SIZE * width * height;
    CUDA_CHECK(cudaMalloc(&a, n_pixel_max * MAX_FILTER_DEPTH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, n_pixel_max * MAX_FILTER_DEPTH * sizeof(float)));

    for (size_t i = 0; i < batches.size(); ++i)
    {
        int n_batch = batches[i].n;

        // First conv2D
        gpu_conv2D(batches[i].get_data(), _encoder_filter_1.get(), a,
                   n_batch, width, height, depth, _ENCODER_FILTER_1_DEPTH);

        // Add bias
        gpu_add_bias(a, _encoder_bias_1.get(), b,
                     n_batch, width, height, _ENCODER_FILTER_1_DEPTH);

        // ReLU
        gpu_relu(b, a, n_batch, width, height, _ENCODER_FILTER_1_DEPTH);

        // Max pooling
        gpu_max_pooling(a, b, n_batch, width, height, _ENCODER_FILTER_1_DEPTH);

        // Second conv2D
        gpu_conv2D(b, _encoder_filter_2.get(), a,
                   n_batch, width / 2, height / 2, _ENCODER_FILTER_1_DEPTH, _ENCODER_FILTER_2_DEPTH);

        gpu_add_bias(a, _encoder_bias_2.get(), b,
                     n_batch, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

        // Second ReLU
        gpu_relu(b, a, n_batch, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

        // Second max pooling
        gpu_max_pooling(a, b, n_batch, width / 2, height / 2, _ENCODER_FILTER_2_DEPTH);

        // Copy batch (FIX Bug 3: Restore full cudaMemcpy and correct size)
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
    // Total elements in the decoded image
    int decoded_image_elements = 4 * width * 4 * height * _DECODER_FILTER_3_DEPTH;
    int offset  = 0;

    vector<Dataset> batches = create_minibatches(dataset, _ENCODE_BATCH_SIZE);
    Dataset res(dataset.n, 4 * width, 4 * height, _DECODER_FILTER_3_DEPTH);

    // Placeholder, alternating - Allocate memory for 4x upsampled output
    float *a, *b;
    int n_pixel_4x_max = _ENCODE_BATCH_SIZE * 4 * width * 4 * height;
    int max_depth_intermediate = std::max(_DECODER_FILTER_3_DEPTH, (int)MAX_FILTER_DEPTH);
    CUDA_CHECK(cudaMalloc(&a, n_pixel_4x_max * max_depth_intermediate * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, n_pixel_4x_max * max_depth_intermediate * sizeof(float)));

    for (size_t i = 0; i < batches.size(); ++i) {
        int n_batch = batches[i].n;

        // First conv2D
        gpu_conv2D(batches[i].get_data(), _decoder_filter_1.get(), a,
                    n_batch, width, height, depth, _DECODER_FILTER_1_DEPTH);

        // Add bias
        gpu_add_bias(a, _decoder_bias_1.get(), b,
                    n_batch, width, height, _DECODER_FILTER_1_DEPTH);

        // ReLU
        gpu_relu(b, a, n_batch, width, height, _DECODER_FILTER_1_DEPTH);

        // Upsampling
        gpu_upsampling(a, b, n_batch, width, height, _DECODER_FILTER_1_DEPTH);

        // Second conv2D
        gpu_conv2D(b, _decoder_filter_2.get(), a,
                    n_batch, 2 * width, 2 * height, _DECODER_FILTER_1_DEPTH, _DECODER_FILTER_2_DEPTH);

        gpu_add_bias(a, _decoder_bias_2.get(), b,
                    n_batch, 2 * width, 2 * height, _DECODER_FILTER_2_DEPTH);

        // Second ReLU
        gpu_relu(b, a, n_batch, 2 * width, 2 * height, _DECODER_FILTER_2_DEPTH);

        // Second upsampling
        gpu_upsampling(a, b, n_batch, 2 * width, 2 * height, _DECODER_FILTER_2_DEPTH);

        // Third conv2D
        gpu_conv2D(b, _decoder_filter_3.get(), a,
                    n_batch, 4 * width, 4 * height, _DECODER_FILTER_2_DEPTH, _DECODER_FILTER_3_DEPTH);

        gpu_add_bias(a, _decoder_bias_3.get(), b,
                    n_batch, 4 * width, 4 * height, _DECODER_FILTER_3_DEPTH);

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
    // Use correct dimensions for decoded output (4*w x 4*h) and depth
    int depth_out = _DECODER_FILTER_3_DEPTH;
    return gpu_mse_loss(dataset.get_data(), decode(encode(dataset)).get_data(),
                        dataset.n, 4 * dataset.width, 4 * dataset.height, depth_out);
}