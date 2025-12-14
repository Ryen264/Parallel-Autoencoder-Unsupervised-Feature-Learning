#include "constants.h"
#include "gpu_autoencoder.h"
#include "gpu_layers.h"
#include "macro.h"
#include "progress_bar.h"
#include "timer.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

using std::ifstream, std::ofstream;
using std::max_element;
using std::memcpy;
using std::printf, std::puts;
using std::stringstream;
using std::swap;

/**
 * @brief Generate a random array with elements between 0 and 1
 *
 * @param arr The array
 * @param n The number of elements
 */
__global__ void generate_array_kernel(float *arr, int n) {
  for (int i = 0; i < n; ++i)
    ptr[i] = 1.0f * rand() / RAND_MAX;
}

/**
 * @brief Generate a random array with elements between 0 and 1
 *
 * @param arr The array
 * @param n The number of elements
 * @param block_size The block size to perform the operation
 */
void generate_array(const Gpu_Unique_Ptr &arr, int n, dim3 block_size) {
  dim3 grid_size((n + block_size.x - 1) / block_size.x + 1);
  generate_array_kernel<<<grid_size, block_size>>>(arr.get(), n);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Read data from a buffer
 *
 * @param buffer The buffer
 * @param data The data
 * @param size Number of bytes to read
 */
void read_data(std::ifstream &buffer, const Gpu_Unique_Ptr &data, int size) {
  unique_ptr<char[]> ptr = make_unique<char[]>(size);
  buffer.read(ptr.get(), size);
  CUDA_CHECK(cudaMemcpy(data.get(), ptr.get(), size, cudaMemcpyHostToDevice));
}

/**
 * @brief Write data to a buffer
 *
 * @param buffer The buffer
 * @param data The data
 * @param size Number of bytes to write
 */
void write_data(std::ostream &buffer, const Gpu_Unique_Ptr &data, int size) {
  unique_ptr<char[]> ptr = make_unique<char[]>(size);
  CUDA_CHECK(cudaMemcpy(ptr.get(), data.get(), size, cudaMemcpyDeviceToHost));
  buffer.write(ptr.get(), size);
}

Gpu_Autoencoder::Gpu_Autoencoder() {
  _allocate_mem();

  // Random init
  srand(time(0));

  generate_array(_encoder_filter_1, ENCODER_FILTER_1_SIZE, _block_size_1D);
  generate_array(_encoder_bias_1, ENCODER_FILTER_1_DEPTH, _block_size_1D);

  generate_array(_encoder_filter_2, ENCODER_FILTER_2_SIZE, _block_size_1D);
  generate_array(_encoder_bias_2, ENCODER_FILTER_2_DEPTH, _block_size_1D);

  generate_array(_decoder_filter_1, DECODER_FILTER_1_SIZE, _block_size_1D);
  generate_array(_decoder_bias_1, DECODER_FILTER_1_DEPTH, _block_size_1D);

  generate_array(_decoder_filter_2, DECODER_FILTER_2_SIZE, _block_size_1D);
  generate_array(_decoder_bias_2, DECODER_FILTER_2_DEPTH, _block_size_1D);

  generate_array(_decoder_filter_2, DECODER_FILTER_2_SIZE, _block_size_1D);
  generate_array(_decoder_bias_2, DECODER_FILTER_2_DEPTH, _block_size_1D);
}

Gpu_Autoencoder::Gpu_Autoencoder(const char *filename) {
  _allocate_mem();

  // Read from file
  ifstream buffer(filename, std::ios::in | std::ios::binary);

  // Read first encoder conv2D layer
  read_data(buffer, _encoder_filter_1, ENCODER_FILTER_1_SIZE * sizeof(float));
  read_data(buffer, _encoder_bias_1, ENCODER_FILTER_1_DEPTH * sizeof(float));

  // Read second encoder conv2D layer
  read_data(buffer, _encoder_filter_2, ENCODER_FILTER_2_SIZE * sizeof(float));
  read_data(buffer, _encoder_bias_2, ENCODER_FILTER_2_DEPTH * sizeof(float));

  // Read first decoder conv2D layer
  read_data(buffer, _decoder_filter_1, DECODER_FILTER_1_SIZE * sizeof(float));
  read_data(buffer, _decoder_bias_1, DECODER_FILTER_1_DEPTH * sizeof(float));

  // Read second decoder conv2D layer
  read_data(buffer, _decoder_filter_2, DECODER_FILTER_2_SIZE * sizeof(float));
  read_data(buffer, _decoder_bias_2, DECODER_FILTER_2_DEPTH * sizeof(float));

  // Read third encoder conv2D layer
  read_data(buffer, _decoder_filter_3, DECODER_FILTER_3_SIZE * sizeof(float));
  read_data(buffer, _decoder_bias_3, DECODER_FILTER_3_DEPTH * sizeof(float));
}

void Gpu_Autoencoder::_allocate_mem() {
  _encoder_filter_1 = Gpu_Unique_Ptr(ENCODER_FILTER_1_SIZE);
  _encoder_bias_1   = Gpu_Unique_Ptr(ENCODER_FILTER_1_DEPTH);

  _encoder_filter_2 = Gpu_Unique_Ptr(ENCODER_FILTER_2_SIZE);
  _encoder_bias_2   = Gpu_Unique_Ptr(ENCODER_FILTER_2_DEPTH);

  _decoder_filter_1 = Gpu_Unique_Ptr(DECODER_FILTER_1_SIZE);
  _decoder_bias_1   = Gpu_Unique_Ptr(DECODER_FILTER_1_DEPTH);

  _decoder_filter_2 = Gpu_Unique_Ptr(DECODER_FILTER_2_SIZE);
  _decoder_bias_2   = Gpu_Unique_Ptr(DECODER_FILTER_2_DEPTH);

  _decoder_filter_3 = Gpu_Unique_Ptr(DECODER_FILTER_3_SIZE);
  _decoder_bias_3   = Gpu_Unique_Ptr(DECODER_FILTER_3_DEPTH);
}

void Gpu_Autoencoder::_forward_pass(const Dataset &dataset) {
  int n = dataset.n, width = dataset.width, height = dataset.height,
      depth = dataset.depth;
  CUDA_CHECK(cudaMemcpy(_batch_data.get(),
                        dataset.get_data(),
                        n * width * height * depth * sizeof(float),
                        cudaMemcpyHostToDevice));

  // First conv2D layer
  gpu_conv2D(_batch_data.get(),
             _encoder_filter_1.get(),
             _out_encoder_filter_1.get(),
             n,
             width,
             height,
             depth,
             ENCODER_FILTER_1_DEPTH,
             _block_size_3D_1);

  // Dim: n * w * w * 256
  gpu_add_bias(_out_encoder_filter_1.get(),
               _encoder_bias_1.get(),
               _out_encoder_bias_1.get(),
               n,
               width,
               height,
               ENCODER_FILTER_1_DEPTH,
               _block_size_1D);

  // ReLU layer
  gpu_relu(_out_encoder_bias_1.get(),
           _out_encoder_relu_1.get(),
           n,
           width,
           height,
           ENCODER_FILTER_1_DEPTH,
           _block_size_1D);

  // First max pooling layer
  gpu_max_pooling(_out_encoder_relu_1.get(),
                  _out_max_pooling_1.get(),
                  n,
                  width,
                  height,
                  ENCODER_FILTER_1_DEPTH,
                  _block_size_3D_2);

  // Dim: n * w/2 * w/2 * 256
  // Second conv2D layer
  gpu_conv2D(_out_max_pooling_1.get(),
             _encoder_filter_2.get(),
             _out_encoder_filter_2.get(),
             n,
             width / 2,
             height / 2,
             ENCODER_FILTER_1_DEPTH,
             ENCODER_FILTER_2_DEPTH,
             _block_size_3D_2);

  // Dim: n * w/2 * w/2 * 128
  gpu_add_bias(_out_encoder_filter_2.get(),
               _encoder_bias_1.get(),
               _out_encoder_bias_2.get(),
               n,
               width / 2,
               height / 2,
               ENCODER_FILTER_2_DEPTH,
               _block_size_1D);

  // ReLU layer
  gpu_relu(_out_encoder_bias_2.get(),
           _out_encoder_relu_2.get(),
           n,
           width / 2,
           height / 2,
           ENCODER_FILTER_2_DEPTH,
           _block_size_1D);

  // Second max pooling layer
  gpu_max_pooling(_out_encoder_relu_2.get(),
                  _out_max_pooling_2.get(),
                  n,
                  width / 2,
                  height / 2,
                  ENCODER_FILTER_2_DEPTH,
                  _block_size_3D_3);

  // First conv2D layer
  gpu_conv2D(_out_max_pooling_2.get(),
             _decoder_filter_1.get(),
             _out_decoder_filter_1.get(),
             n,
             width / 4,
             height / 4,
             ENCODER_FILTER_2_DEPTH,
             DECODER_FILTER_1_DEPTH,
             _block_size_3D_3);

  // Dim: n * w * w * 128
  gpu_add_bias(_out_decoder_filter_1.get(),
               _decoder_bias_1.get(),
               _out_decoder_bias_1.get(),
               n,
               width / 4,
               height / 4,
               DECODER_FILTER_1_DEPTH,
               _block_size_1D);

  // ReLU layer
  gpu_relu(_out_decoder_bias_1.get(),
           _out_decoder_relu_1.get(),
           n,
           width / 4,
           height / 4,
           DECODER_FILTER_1_DEPTH,
           _block_size_1D);

  // First upsampling layer
  gpu_upsampling(_out_decoder_relu_1.get(),
                 _out_upsampling_1.get(),
                 n,
                 width / 4,
                 height / 4,
                 DECODER_FILTER_1_DEPTH,
                 _block_size_3D_3);

  // Dim: n * 2w * 2w * 256
  // Second conv2D layer
  gpu_conv2D(_out_upsampling_1.get(),
             _decoder_filter_2.get(),
             _out_decoder_filter_2.get(),
             n,
             width / 2,
             height / 2,
             DECODER_FILTER_1_DEPTH,
             DECODER_FILTER_2_DEPTH,
             _block_size_3D_2);

  // Dim: n * 2w * 2w * 256
  gpu_add_bias(_out_decoder_filter_2.get(),
               _decoder_bias_2.get(),
               _out_decoder_bias_2.get(),
               n,
               width / 2,
               height / 2,
               DECODER_FILTER_2_DEPTH,
               _block_size_1D);

  // ReLU layer
  gpu_relu(_out_decoder_bias_2.get(),
           _out_decoder_relu_2.get(),
           n,
           width / 2,
           height / 2,
           DECODER_FILTER_2_DEPTH,
           _block_size_1D);

  // Second upsampling layer
  gpu_upsampling(_out_decoder_relu_2.get(),
                 _out_upsampling_2.get(),
                 n,
                 width / 2,
                 height / 2,
                 DECODER_FILTER_2_DEPTH,
                 _block_size_3D_2);

  // Dim: n * 4w * 4w * 256
  // Third conv2D layer
  gpu_conv2D(_out_upsampling_2.get(),
             _decoder_filter_3.get(),
             _out_decoder_filter_3.get(),
             n,
             width,
             height,
             DECODER_FILTER_2_DEPTH,
             DECODER_FILTER_3_DEPTH,
             _block_size_3D_1);

  // Dim: n * 4w * 4w * 3
  gpu_add_bias(_out_decoder_filter_3.get(),
               _decoder_bias_3.get(),
               _out_decoder_bias_3.get(),
               n,
               width,
               height,
               DECODER_FILTER_3_DEPTH,
               _block_size_1D);

  // Return the result (Dim: n * w/4 * w/4 * 128)
  CUDA_CHECK(
      cudaMemcpy(_res_data.get(),
                 _out_decoder_bias_3.get(),
                 n * 4 * width * 4 * height * DECODER_FILTER_3_DEPTH * sizeof(float),
                 cudaMemcpyDeviceToDevice));
}

void Gpu_Autoencoder::_allocate_output_mem(int n, int width, int height) {
  int n_pixel = n * width * height;

  _out_encoder_filter_1 = Gpu_Unique_Ptr(n_pixel * ENCODER_FILTER_1_DEPTH);
  _out_encoder_bias_1   = Gpu_Unique_Ptr(n_pixel * ENCODER_FILTER_1_DEPTH);
  _out_encoder_relu_1   = Gpu_Unique_Ptr(n_pixel * ENCODER_FILTER_1_DEPTH);
  _out_max_pooling_1    = Gpu_Unique_Ptr(n_pixel * ENCODER_FILTER_1_DEPTH / 4);

  _out_encoder_filter_2 = Gpu_Unique_Ptr(n_pixel * ENCODER_FILTER_2_DEPTH);
  _out_encoder_bias_2   = Gpu_Unique_Ptr(n_pixel * ENCODER_FILTER_2_DEPTH);
  _out_encoder_relu_2   = Gpu_Unique_Ptr(n_pixel * ENCODER_FILTER_2_DEPTH);
  _out_max_pooling_2    = Gpu_Unique_Ptr(n_pixel * ENCODER_FILTER_2_DEPTH / 16);

  _out_decoder_filter_1 = Gpu_Unique_Ptr(n_pixel * DECODER_FILTER_1_DEPTH / 16);
  _out_decoder_bias_1   = Gpu_Unique_Ptr(n_pixel * DECODER_FILTER_1_DEPTH / 16);
  _out_decoder_relu_1   = Gpu_Unique_Ptr(n_pixel * DECODER_FILTER_1_DEPTH / 16);
  _out_upsampling_1     = Gpu_Unique_Ptr(n_pixel * DECODER_FILTER_1_DEPTH / 4);

  _out_decoder_filter_2 = Gpu_Unique_Ptr(n_pixel * DECODER_FILTER_2_DEPTH / 4);
  _out_decoder_bias_2   = Gpu_Unique_Ptr(n_pixel * DECODER_FILTER_2_DEPTH / 4);
  _out_decoder_relu_2   = Gpu_Unique_Ptr(n_pixel * DECODER_FILTER_2_DEPTH / 4);
  _out_upsampling_2     = Gpu_Unique_Ptr(n_pixel * DECODER_FILTER_2_DEPTH);

  _out_decoder_filter_3 = Gpu_Unique_Ptr(n_pixel * DECODER_FILTER_3_DEPTH);
  _out_decoder_bias_3   = Gpu_Unique_Ptr(n_pixel * DECODER_FILTER_3_DEPTH);

  _batch_data = Gpu_Unique_Ptr(n_pixel * IMAGE_DEPTH);
  _res_data   = Gpu_Unique_Ptr(n_pixel * IMAGE_DEPTH);

  static constexpr int FILTER_SIZES[]  = { ENCODER_FILTER_1_SIZE,
                                           ENCODER_FILTER_2_SIZE,
                                           DECODER_FILTER_1_SIZE,
                                           DECODER_FILTER_2_SIZE,
                                           DECODER_FILTER_3_SIZE };
  constexpr int        MAX_FILTER_SIZE = *max_element(FILTER_SIZES, FILTER_SIZES + 5);

  _d_in     = Gpu_Unique_Ptr(MAX_FILTER_DEPTH);
  _d_out    = Gpu_Unique_Ptr(n * width * height * MAX_FILTER_DEPTH);
  _d_filter = Gpu_Unique_Ptr(MAX_FILTER_SIZE);
}

void Gpu_Autoencoder::_deallocate_output_mem() {
  _out_encoder_filter_1.reset();
  _out_encoder_bias_1.reset();
  _out_encoder_relu_1.reset();
  _out_max_pooling_1.reset();

  _out_encoder_filter_2.reset();
  _out_encoder_bias_2.reset();
  _out_encoder_relu_2.reset();
  _out_max_pooling_2.reset();

  _out_decoder_filter_1.reset();
  _out_decoder_bias_1.reset();
  _out_decoder_relu_1.reset();
  _out_upsampling_1.reset();

  _out_decoder_filter_2.reset();
  _out_decoder_bias_2.reset();
  _out_decoder_relu_2.reset();
  _out_upsampling_2.reset();

  _out_decoder_filter_3.reset();
  _out_decoder_bias_3.reset();

  _batch_data.reset();
  _res_data.reset();

  _d_in.reset();
  _d_out.reset();
  _d_filter.reset();
}

float Gpu_Autoencoder::_fit_batch(const Dataset &batch, float learning_rate) {
  // Get the result after autoencoding
  int    n = batch.n, width = batch.width, height = batch.height, depth = batch.depth;
  float *d_in = _d_in.get(), *d_out = _d_out.get(), *d_filter = _d_filter.get();
  _forward_pass(batch);

  // Calculate loss before backprop
  float loss = gpu_mse_loss(
      _batch_data.get(), _res_data.get_data(), n, width, height, depth, _block_size_1D);

  // Get loss gradient
  gpu_mse_grad(_batch_data.get(),
               _res_data.get(),
               d_out,
               n,
               width,
               height,
               depth,
               _block_size_1D);

  // Update weight for the last conv2D layer
  // Update bias
  gpu_bias_grad(d_out, d_in, n, width, height, DECODER_FILTER_3_DEPTH, _block_size_1D);

  gpu_update_weight(_decoder_bias_3.get(),
                    d_in,
                    DECODER_FILTER_3_DEPTH,
                    learning_rate,
                    _block_size_1D);

  // Update filter
  gpu_conv2D_grad(_out_upsampling_2.get(),
                  d_out,
                  d_filter,
                  n,
                  width,
                  height,
                  DECODER_FILTER_2_DEPTH,
                  DECODER_FILTER_3_DEPTH,
                  _block_size_3D_1);

  // Pass delta backwards
  gpu_conv2D(d_out,
             _decoder_filter_3.get(),
             d_in,
             n,
             width,
             height,
             DECODER_FILTER_2_DEPTH,
             DECODER_FILTER_3_DEPTH,
             _block_size_3D_1);

  // Swap d_out and d_in
  swap(d_out, d_in);

  // Update weight
  gpu_update_weight(_decoder_filter_3.get(),
                    d_filter,
                    DECODER_FILTER_3_SIZE,
                    learning_rate,
                    _block_size_1D);

  // Pass through upsampling (dim: n * w/2 * w/2 * 256)
  gpu_upsampling_backward(
      d_out, d_in, n, width / 2, height / 2, DECODER_FILTER_2_DEPTH, _block_size_3D_2);

  // Pass through ReLU (d_in and d_out swapped)
  gpu_relu_backward(_out_decoder_bias_2.get(),
                    d_in,
                    d_out,
                    n,
                    width / 2,
                    height / 2,
                    DECODER_FILTER_2_DEPTH,
                    _block_size_1D);

  // Second conv2D layer
  gpu_bias_grad(
      d_out, d_in, n, width / 2, height / 2, DECODER_FILTER_2_DEPTH, _block_size_1D);

  gpu_update_weight(_decoder_bias_2.get(),
                    d_in,
                    DECODER_FILTER_2_DEPTH,
                    learning_rate,
                    _block_size_1D);

  gpu_conv2D_grad(_out_upsampling_1.get(),
                  d_out,
                  d_filter,
                  n,
                  width / 2,
                  height / 2,
                  DECODER_FILTER_1_DEPTH,
                  DECODER_FILTER_2_DEPTH,
                  _block_size_3D_2);

  gpu_conv2D(d_out,
             _decoder_filter_2.get(),
             d_in,
             n,
             width / 2,
             height / 2,
             DECODER_FILTER_1_DEPTH,
             DECODER_FILTER_2_DEPTH,
             _block_size_3D_2);

  swap(d_out, d_in);
  gpu_update_weight(_decoder_filter_2.get(),
                    d_filter,
                    DECODER_FILTER_2_SIZE,
                    learning_rate,
                    _block_size_1D);

  // Upsampling (dim: n * w/4 * w/4 * 128)
  gpu_upsampling_backward(
      d_out, d_in, n, width / 4, height / 4, DECODER_FILTER_1_DEPTH, _block_size_3D_3);

  // ReLU
  gpu_relu_backward(_out_decoder_bias_1.get(),
                    d_in,
                    d_out,
                    n,
                    width / 4,
                    height / 4,
                    DECODER_FILTER_1_DEPTH,
                    _block_size_1D);

  // Third Conv2D
  gpu_bias_grad(
      d_out, d_in, n, width / 4, height / 4, DECODER_FILTER_1_DEPTH, _block_size_1D);

  gpu_update_weight(_decoder_bias_1.get(),
                    d_in,
                    DECODER_FILTER_1_DEPTH,
                    learning_rate,
                    _block_size_1D);

  gpu_conv2D_grad(_out_max_pooling_2.get(),
                  d_out,
                  d_filter,
                  n,
                  width / 4,
                  height / 4,
                  ENCODER_FILTER_2_DEPTH,
                  DECODER_FILTER_1_DEPTH,
                  _block_size_3D_3);

  gpu_conv2D(d_out,
             _decoder_filter_1.get(),
             d_in,
             n,
             width / 4,
             height / 4,
             ENCODER_FILTER_2_DEPTH,
             DECODER_FILTER_1_DEPTH,
             _block_size_3D_3);

  swap(d_out, d_in);
  gpu_update_weight(_decoder_filter_1.get(),
                    d_filter,
                    DECODER_FILTER_1_SIZE,
                    learning_rate,
                    _block_size_1D);

  // Max pooling backwards (dim: n * w/2 * w/2 * 128)
  gpu_max_pooling_backward(_out_encoder_relu_2.get(),
                           d_out,
                           d_in,
                           n,
                           width / 2,
                           height / 2,
                           ENCODER_FILTER_2_DEPTH,
                           _block_size_3D_2);

  gpu_relu_backward(_out_encoder_bias_2.get(),
                    d_in,
                    d_out,
                    n,
                    width / 2,
                    height / 2,
                    ENCODER_FILTER_2_DEPTH,
                    _block_size_1D);

  // Forth conv2D
  gpu_bias_grad(
      d_out, d_in, n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH, _block_size_1D);

  gpu_update_weight(_encoder_bias_2.get(),
                    d_in,
                    ENCODER_FILTER_2_DEPTH,
                    learning_rate,
                    _block_size_1D);

  gpu_conv2D_grad(_out_max_pooling_1.get(),
                  d_out,
                  d_filter,
                  n,
                  width / 2,
                  height / 2,
                  ENCODER_FILTER_1_DEPTH,
                  ENCODER_FILTER_2_DEPTH,
                  _block_size_3D_2);

  gpu_conv2D(d_out,
             _encoder_filter_2.get(),
             d_in,
             n,
             width / 2,
             height / 2,
             ENCODER_FILTER_1_DEPTH,
             ENCODER_FILTER_2_DEPTH,
             _block_size_3D_2);

  swap(d_out, d_in);
  gpu_update_weight(_encoder_filter_2.get(),
                    d_filter,
                    ENCODER_FILTER_2_SIZE,
                    learning_rate,
                    _block_size_1D);

  gpu_max_pooling_backward(_out_encoder_relu_1.get(),
                           d_out,
                           d_in,
                           n,
                           width,
                           height,
                           ENCODER_FILTER_1_DEPTH,
                           _block_size_3D_1);

  gpu_relu_backward(_out_encoder_bias_1.get(),
                    d_in,
                    d_out,
                    n,
                    width,
                    height,
                    ENCODER_FILTER_1_DEPTH,
                    _block_size_3D_1);

  // Fifth conv2D
  gpu_bias_grad(d_out, d_in, n, width, height, ENCODER_FILTER_1_DEPTH, _block_size_1D);

  gpu_update_weight(_encoder_bias_1.get(),
                    d_in,
                    ENCODER_FILTER_1_DEPTH,
                    learning_rate,
                    _block_size_1D);

  gpu_conv2D_grad(batch.get_data(),
                  d_out,
                  d_filter,
                  n,
                  width,
                  height,
                  depth,
                  ENCODER_FILTER_1_DEPTH,
                  _block_size_3D_1);

  gpu_conv2D(d_out,
             _encoder_filter_1.get(),
             d_in,
             n,
             width,
             height,
             depth,
             ENCODER_FILTER_1_DEPTH,
             _block_size_3D_1);

  swap(d_out, d_in);
  gpu_update_weight(_encoder_filter_1.get(),
                    d_filter,
                    ENCODER_FILTER_1_SIZE,
                    learning_rate,
                    _block_size_1D);

  return loss;
}

void Gpu_Autoencoder::fit(const Dataset &dataset,
                          int            n_epoch,
                          int            batch_size,
                          float          learning_rate,
                          bool           verbose,
                          int            checkpoint,
                          const char    *output_dir) {
  // Create minibatches
  vector<Dataset> batches = create_minibatches(dataset, batch_size);

  // Allocate memory for training
  _allocate_output_mem(batch_size, dataset.width, dataset.height);

  Timer        timer;
  Progress_Bar bar(batches.size(), "Batch");
  float        total_time = 0;

  puts("=======================TRAINING START=======================");
  for (int epoch = 1; epoch <= n_epoch; ++epoch) {
    bar.update();
    timer.start();

    float total_loss = 0;
    for (const Dataset &batch : batches) {
      total_loss += _fit_batch(batch, learning_rate) * batch.n;
    }

    timer.stop();
    float epoch_time  = timer.get();
    total_time       += epoch_time;

    // Print average loss for the epoch
    float avg_loss = total_loss / dataset.n;
    printf("Time: %.2f (ms), Loss: %.4f\n", epoch_time, avg_loss);

    // Save at checkpoints
    if (checkpoint > 0 && epoch % checkpoint == 0) {
      stringstream builder;
      builder << output_dir << '/' << "gpu_autoencoder_" << epoch << ".bin";
      save_parameters(builder.str().c_str());
    }
  }

  puts("========================TRAINING END========================");

  // Deallocate memory to remove unused memory
  _deallocate_output_mem();

  // Save models param
  stringstream builder;
  builder << output_dir << '/' << "gpu_autoencoder_.bin";
  save_parameters(builder.str().c_str());

  printf("\nTotal time: %.2f (ms)", total_time);
}

Dataset Gpu_Autoencoder::encode(const Dataset &dataset) const {
  // Encode by batches to use less memory
  int width = dataset.width, height = dataset.height, depth = dataset.depth;
  int image_bytes = width * height * depth * sizeof(float);
  int encoded_image_bytes =
      width / 4 * height / 4 * ENCODER_FILTER_2_DEPTH * sizeof(float);
  int offset = 0;

  vector<Dataset> batches = create_minibatches(dataset, ENCODE_BATCH_SIZE);
  Dataset         res(dataset.n, width / 4, height / 4, ENCODER_FILTER_2_DEPTH);

  // Placeholder, alternating
  Gpu_Unique_Ptr a(ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH);
  Gpu_Unique_Ptr b(ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH);

  for (int i = 0; i < batches.size(); ++i) {
    int n = batches[i].n;
    CUDA_CHECK(cudaMemcpy(
        b.get(), batches[i].get_data(), n * image_bytes, cudaMemcpyHostToDevice));

    // First conv2D
    gpu_conv2D(b.get(),
               _encoder_filter_1.get(),
               a.get(),
               n,
               width,
               height,
               depth,
               ENCODER_FILTER_1_DEPTH,
               _block_size_3D_1);

    // Add bias
    gpu_add_bias(a.get(),
                 _encoder_bias_1.get(),
                 b.get(),
                 n,
                 width,
                 height,
                 ENCODER_FILTER_1_DEPTH,
                 _block_size_1D);

    // ReLU
    gpu_relu(
        b.get(), a.get(), n, width, height, ENCODER_FILTER_1_DEPTH, _block_size_1D);

    // Max pooling
    gpu_max_pooling(
        a.get(), b.get(), n, width, height, ENCODER_FILTER_1_DEPTH, _block_size_3D_2);

    // Second conv2D
    gpu_conv2D(b.get(),
               _encoder_filter_2.get(),
               a.get(),
               n,
               width / 2,
               height / 2,
               ENCODER_FILTER_1_DEPTH,
               ENCODER_FILTER_2_DEPTH,
               _block_size_3D_2);

    gpu_add_bias(a.get(),
                 _encoder_bias_2.get(),
                 b.get(),
                 n,
                 width / 2,
                 height / 2,
                 ENCODER_FILTER_2_DEPTH,
                 _block_size_1D);

    // Second ReLU
    gpu_relu(b.get(),
             a.get(),
             n,
             width / 2,
             height / 2,
             ENCODER_FILTER_2_DEPTH,
             _block_size_1D);

    // Second max pooling
    gpu_max_pooling(a.get(),
                    b.get(),
                    n,
                    width / 2,
                    height / 2,
                    ENCODER_FILTER_2_DEPTH,
                    _block_size_3D_3);

    // Copy batch
    CUDA_CHECK(cudaMemcpy(res.get_data() + offset,
                          b.get(),
                          n * encoded_image_bytes,
                          cudaMemcpyDeviceToHost));
    offset += n * encoded_image_bytes;
  }

  // Copy labels
  memcpy(res.get_labels(), dataset.get_labels(), dataset.n * sizeof(int));
  return res;
}

Dataset Gpu_Autoencoder::decode(const Dataset &dataset) const {
  int width = dataset.width, height = dataset.height, depth = dataset.depth;
  int image_bytes = width * height * depth * sizeof(float);
  int decoded_image_bytes =
      4 * width * 4 * height * DECODER_FILTER_3_DEPTH * sizeof(float);
  int offset = 0;

  vector<Dataset> batches = create_minibatches(dataset, ENCODE_BATCH_SIZE);
  Dataset         res(dataset.n, width * 4, height * 4, DECODER_FILTER_3_DEPTH);

  // Placeholder, alternating
  Gpu_Unique_Ptr a(ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH);
  Gpu_Unique_Ptr b(ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH);

  for (int i = 0; i < batches.size(); ++i) {
    int n = batches[i].n;
    CUDA_CHECK(cudaMemcpy(
        b.get(), batches[i].get_data(), n * image_bytes, cudaMemcpyHostToDevice));

    // First conv2D
    gpu_conv2D(b.get(),
               _decoder_filter_1.get(),
               a.get(),
               n,
               width,
               height,
               depth,
               DECODER_FILTER_1_DEPTH,
               _block_size_3D_3);

    // Add bias
    gpu_add_bias(a.get(),
                 _decoder_bias_1.get(),
                 b.get(),
                 n,
                 width,
                 height,
                 DECODER_FILTER_1_DEPTH,
                 _block_size_1D);

    // ReLU
    gpu_relu(
        b.get(), a.get(), n, width, height, DECODER_FILTER_1_DEPTH, _block_size_1D);

    // Upsampling
    gpu_upsampling(
        a.get(), b.get(), n, width, height, DECODER_FILTER_1_DEPTH, _block_size_3D_3);

    // Second conv2D
    gpu_conv2D(b.get(),
               _decoder_filter_2.get(),
               a.get(),
               n,
               width * 2,
               height * 2,
               DECODER_FILTER_1_DEPTH,
               DECODER_FILTER_2_DEPTH,
               _block_size_3D_2);

    gpu_add_bias(a.get(),
                 _decoder_bias_2.get(),
                 b.get(),
                 n,
                 width * 2,
                 height * 2,
                 DECODER_FILTER_2_DEPTH,
                 _block_size_1D);

    // Second ReLU
    gpu_relu(b.get(),
             a.get(),
             n,
             width * 2,
             height * 2,
             DECODER_FILTER_2_DEPTH,
             _block_size_1D);

    // Second upsampling
    gpu_upsampling(a.get(),
                   b.get(),
                   n,
                   width * 2,
                   height * 2,
                   DECODER_FILTER_2_DEPTH,
                   _block_size_3D_2);

    // Third conv2D
    gpu_conv2D(b.get(),
               _decoder_filter_3.get(),
               a.get(),
               n,
               width * 4,
               height * 4,
               DECODER_FILTER_2_DEPTH,
               DECODER_FILTER_3_DEPTH,
               _block_size_3D_1);

    gpu_add_bias(a.get(),
                 _decoder_bias_3.get(),
                 b.get(),
                 n,
                 width * 4,
                 height * 4,
                 DECODER_FILTER_3_DEPTH,
                 _block_size_1D);

    // Copy batch
    CUDA_CHECK(cudaMemcpy(res.get_data() + offset,
                          b.get(),
                          n * decoded_image_bytes,
                          cudaMemcpyDeviceToHost));
    offset += n * decoded_image_bytes;
  }

  // Copy the result
  memcpy(res.get_labels(), dataset.get_labels(), dataset.n * sizeof(int));
  return res;
}

float Gpu_Autoencoder::eval(const Dataset &dataset) const {
  int n = dataset.n, width = dataset.width, height = dataset.height,
      depth    = dataset.depth;
  int     size = n * width * height * depth;
  Dataset res  = decode(encode(dataset));

  Gpu_Unique_Ptr a(size);
  Gpu_Unique_Ptr b(size);
  CUDA_CHECK(
      cudaMemcpy(a.get(), dataset.get_data(), size * sizeof(float), cudaMemcpyHost));
  CUDA_CHECK(cudaMemcpy(b.get(), res.get_data(), size * sizeof(float), cudaMemcpyHost));
  return gpu_mse_loss(a.get(), b.get(), n, width, height, depth, _block_size_1D);
}

void Gpu_Autoencoder::save_parameters(const char *filename) const {
  ofstream buffer(filename, std::ios::out | std::ios::binary);

  // Write first encoder conv2D layer
  write_data(buffer, _encoder_filter_1, ENCODER_FILTER_1_SIZE * sizeof(float));
  write_data(buffer, _encoder_bias_1, ENCODER_FILTER_1_DEPTH * sizeof(float));

  // Write second encoder conv2D layer
  write_data(buffer, _encoder_filter_2, ENCODER_FILTER_2_SIZE * sizeof(float));
  write_data(buffer, _encoder_bias_2, ENCODER_FILTER_2_DEPTH * sizeof(float));

  // Write first decoder conv2D layer
  write_data(buffer, _decoder_filter_1, DECODER_FILTER_1_SIZE * sizeof(float));
  write_data(buffer, _decoder_bias_1, DECODER_FILTER_1_DEPTH * sizeof(float));

  // Write second decoder conv2D layer
  write_data(buffer, _decoder_filter_2, DECODER_FILTER_2_SIZE * sizeof(float));
  write_data(buffer, _decoder_bias_2, DECODER_FILTER_2_DEPTH * sizeof(float));

  // Write third encoder conv2D layer
  write_data(buffer, _decoder_filter_3, DECODER_FILTER_3_SIZE * sizeof(float));
  write_data(buffer, _decoder_bias_3, DECODER_FILTER_3_DEPTH * sizeof(float));
}