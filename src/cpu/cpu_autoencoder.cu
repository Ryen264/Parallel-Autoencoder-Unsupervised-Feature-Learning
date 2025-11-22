#include "constants.h"
#include "cpu_autoencoder.h"
#include "cpu_layers.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

using std::max_element;
using std::memcpy;
using std::printf, std::puts;
using std::swap;

Cpu_Autoencoder::Cpu_Autoencoder()
    : IAutoencoder() {};

Cpu_Autoencoder::Cpu_Autoencoder(const char *filename)
    : IAutoencoder(filename) {};

Dataset Cpu_Autoencoder::_encode_save_output(const Dataset &dataset) {
  int n     = dataset.n;
  int width = dataset.width;

  // First conv2D layer
  cpu_conv2D(dataset.get_data(),
             _encoder_filter_1.get(),
             _out_encoder_filter_1.get(),
             n,
             width,
             dataset.depth,
             _ENCODER_FILTER_1_DEPTH);
  // Dim: n * w * w * 256
  cpu_add_bias(_out_encoder_filter_1.get(),
               _encoder_bias_1.get(),
               _out_encoder_bias_1.get(),
               n,
               width,
               _ENCODER_FILTER_1_DEPTH);

  // ReLU layer
  cpu_relu(_out_encoder_bias_1.get(),
           _out_encoder_relu_1.get(),
           n,
           width,
           _ENCODER_FILTER_1_DEPTH);

  // First max pooling layer
  cpu_max_pooling(_out_encoder_relu_1.get(),
                  _out_max_pooling_1.get(),
                  n,
                  width,
                  _ENCODER_FILTER_1_DEPTH);

  // Dim: n * w/2 * w/2 * 256
  // Second conv2D layer
  cpu_conv2D(_out_max_pooling_1.get(),
             _encoder_filter_2.get(),
             _out_encoder_filter_2.get(),
             n,
             width / 2,
             _ENCODER_FILTER_1_DEPTH,
             _ENCODER_FILTER_2_DEPTH);
  // Dim: n * w/2 * w/2 * 128
  cpu_add_bias(_out_encoder_filter_2.get(),
               _encoder_bias_1.get(),
               _out_encoder_bias_2.get(),
               n,
               width / 2,
               _ENCODER_FILTER_2_DEPTH);

  // ReLU layer
  cpu_relu(_out_encoder_bias_2.get(),
           _out_encoder_relu_2.get(),
           n,
           width / 2,
           _ENCODER_FILTER_2_DEPTH);

  // Second max pooling layer
  cpu_max_pooling(_out_encoder_relu_2.get(),
                  _out_max_pooling_2.get(),
                  n,
                  width / 2,
                  _ENCODER_FILTER_2_DEPTH);

  // Return the result (Dim: n * w/4 * w/4 * 128)
  Dataset res(n, width / 4, _ENCODER_FILTER_2_DEPTH);
  memcpy(res.get_data(),
         _out_max_pooling_2.get(),
         n * width * width * _ENCODER_FILTER_2_DEPTH * sizeof(float) / 16);
  memcpy(res.get_labels(), dataset.get_labels(), n * sizeof(int));

  return res;
}

Dataset Cpu_Autoencoder::_decode_save_output(const Dataset &dataset) {
  int n     = dataset.n;
  int width = dataset.width;

  // First conv2D layer
  cpu_conv2D(dataset.get_data(),
             _decoder_filter_1.get(),
             _out_decoder_filter_1.get(),
             n,
             width,
             dataset.depth,
             _DECODER_FILTER_1_DEPTH);
  // Dim: n * w * w * 128
  cpu_add_bias(_out_decoder_filter_1.get(),
               _encoder_bias_1.get(),
               _out_decoder_bias_1.get(),
               n,
               width,
               _DECODER_FILTER_1_DEPTH);

  // ReLU layer
  cpu_relu(_out_decoder_bias_1.get(),
           _out_decoder_relu_1.get(),
           n,
           width,
           _DECODER_FILTER_1_DEPTH);

  // First upsampling layer
  cpu_upsampling(_out_decoder_relu_1.get(),
                 _out_upsampling_1.get(),
                 n,
                 width,
                 _DECODER_FILTER_1_DEPTH);

  // Dim: n * 2w * 2w * 256
  // Second conv2D layer
  cpu_conv2D(_out_upsampling_1.get(),
             _decoder_filter_2.get(),
             _out_decoder_filter_2.get(),
             n,
             2 * width,
             _DECODER_FILTER_1_DEPTH,
             _DECODER_FILTER_2_DEPTH);
  // Dim: n * 2w * 2w * 256
  cpu_add_bias(_out_decoder_filter_2.get(),
               _decoder_bias_2.get(),
               _out_decoder_bias_2.get(),
               n,
               2 * width,
               _DECODER_FILTER_2_DEPTH);

  // ReLU layer
  cpu_relu(_out_decoder_bias_2.get(),
           _out_decoder_relu_2.get(),
           n,
           2 * width,
           _DECODER_FILTER_2_DEPTH);

  // Second upsampling layer
  cpu_upsampling(_out_decoder_relu_2.get(),
                 _out_upsampling_2.get(),
                 n,
                 2 * width,
                 _DECODER_FILTER_2_DEPTH);

  // Dim: n * 4w * 4w * 256
  // Third conv2D layer
  cpu_conv2D(_out_upsampling_2.get(),
             _decoder_filter_3.get(),
             _out_decoder_filter_3.get(),
             n,
             4 * width,
             _DECODER_FILTER_2_DEPTH,
             _DECODER_FILTER_3_DEPTH);
  // Dim: n * 4w * 4w * 3
  cpu_add_bias(_out_decoder_filter_3.get(),
               _decoder_bias_3.get(),
               _out_decoder_bias_3.get(),
               n,
               4 * width,
               _DECODER_FILTER_3_DEPTH);

  // Return the result (Dim: n * w/4 * w/4 * 128)
  Dataset res(n, width / 4, _ENCODER_FILTER_2_DEPTH);
  memcpy(res.get_data(),
         _out_decoder_bias_3.get(),
         n * width * width * _DECODER_FILTER_3_DEPTH * 16 * sizeof(float));
  memcpy(res.get_labels(), dataset.get_labels(), n * sizeof(int));

  return res;
}

void Cpu_Autoencoder::_allocate_output_mem(int n, int width) {
  int n_pixel = n * width * width;

  _out_encoder_filter_1 = make_unique<float[]>(n_pixel * _ENCODER_FILTER_1_DEPTH);
  _out_encoder_bias_1   = make_unique<float[]>(n_pixel * _ENCODER_FILTER_1_DEPTH);
  _out_encoder_relu_1   = make_unique<float[]>(n_pixel * _ENCODER_FILTER_1_DEPTH);
  _out_max_pooling_1    = make_unique<float[]>(n_pixel * _ENCODER_FILTER_1_DEPTH / 4);

  _out_encoder_filter_2 = make_unique<float[]>(n_pixel * _ENCODER_FILTER_2_DEPTH);
  _out_encoder_bias_2   = make_unique<float[]>(n_pixel * _ENCODER_FILTER_2_DEPTH);
  _out_encoder_relu_2   = make_unique<float[]>(n_pixel * _ENCODER_FILTER_2_DEPTH);
  _out_max_pooling_2    = make_unique<float[]>(n_pixel * _ENCODER_FILTER_2_DEPTH / 16);

  _out_decoder_filter_1 = make_unique<float[]>(n_pixel * _DECODER_FILTER_1_DEPTH / 16);
  _out_decoder_bias_1   = make_unique<float[]>(n_pixel * _DECODER_FILTER_1_DEPTH / 16);
  _out_decoder_relu_1   = make_unique<float[]>(n_pixel * _DECODER_FILTER_1_DEPTH / 16);
  _out_upsampling_1     = make_unique<float[]>(n_pixel * _DECODER_FILTER_1_DEPTH / 4);

  _out_decoder_filter_2 = make_unique<float[]>(n_pixel * _DECODER_FILTER_2_DEPTH / 4);
  _out_decoder_bias_2   = make_unique<float[]>(n_pixel * _DECODER_FILTER_2_DEPTH / 4);
  _out_decoder_relu_2   = make_unique<float[]>(n_pixel * _DECODER_FILTER_2_DEPTH / 4);
  _out_upsampling_2     = make_unique<float[]>(n_pixel * _DECODER_FILTER_2_DEPTH);

  _out_decoder_filter_3 = make_unique<float[]>(n_pixel * _DECODER_FILTER_3_DEPTH);
  _out_decoder_bias_3   = make_unique<float[]>(n_pixel * _DECODER_FILTER_3_DEPTH);

  static constexpr int FILTER_SIZES[]  = { _ENCODER_FILTER_1_SIZE,
                                           _ENCODER_FILTER_2_SIZE,
                                           _DECODER_FILTER_1_SIZE,
                                           _DECODER_FILTER_2_SIZE,
                                           _DECODER_FILTER_3_SIZE };
  constexpr int        MAX_FILTER_SIZE = *max_element(FILTER_SIZES, FILTER_SIZES + 5);

  _d_in     = make_unique<float[]>(MAX_FILTER_DEPTH);
  _d_out    = make_unique<float[]>(n * width * width * MAX_FILTER_DEPTH);
  _d_filter = make_unique<float[]>(MAX_FILTER_SIZE);
}

void Cpu_Autoencoder::_deallocate_output_mem() {
  _out_encoder_filter_1 = 0;
  _out_encoder_bias_1   = 0;
  _out_encoder_relu_1   = 0;
  _out_max_pooling_1    = 0;

  _out_encoder_filter_2 = 0;
  _out_encoder_bias_2   = 0;
  _out_encoder_relu_2   = 0;
  _out_max_pooling_2    = 0;

  _out_decoder_filter_1 = 0;
  _out_decoder_bias_1   = 0;
  _out_decoder_relu_1   = 0;
  _out_upsampling_1     = 0;

  _out_decoder_filter_2 = 0;
  _out_decoder_bias_2   = 0;
  _out_decoder_relu_2   = 0;
  _out_upsampling_2     = 0;

  _out_decoder_filter_3 = 0;
  _out_decoder_bias_3   = 0;

  _d_in     = 0;
  _d_out    = 0;
  _d_filter = 0;
}

float Cpu_Autoencoder::_fit_batch(const Dataset &batch, float learning_rate) {
  // Get the result after autoencoding
  int     n        = batch.n;
  int     width    = batch.width;
  int     depth    = batch.depth;
  float  *d_in     = _d_in.get();
  float  *d_out    = _d_out.get();
  float  *d_filter = _d_filter.get();
  Dataset res      = _decode_save_output(_encode_save_output(batch));

  // Get loss gradient
  cpu_mse_grad(batch.get_data(), res.get_data(), d_out, n, width, depth);

  // Update weight for the last conv2D layer
  // Update bias
  cpu_bias_grad(d_out, d_in, n, width, _DECODER_FILTER_3_DEPTH);
  cpu_update_weight(
      _decoder_bias_3.get(), d_in, _DECODER_FILTER_3_DEPTH, learning_rate);
  // Update filter
  cpu_conv2D_grad(_out_upsampling_2.get(),
                  d_out,
                  d_filter,
                  n,
                  width,
                  _DECODER_FILTER_2_DEPTH,
                  _DECODER_FILTER_3_DEPTH);
  // Pass delta backwards
  cpu_conv2D(d_out,
             _decoder_filter_3.get(),
             d_in,
             n,
             width,
             _DECODER_FILTER_2_DEPTH,
             _DECODER_FILTER_3_DEPTH);
  // Swap d_out and d_in
  swap(d_out, d_in);
  // Update weight
  cpu_update_weight(
      _decoder_filter_3.get(), d_filter, _DECODER_FILTER_3_SIZE, learning_rate);

  // Pass through upsampling (dim: n * w/2 * w/2 * 256)
  cpu_upsampling_backward(d_out, d_in, n, width / 2, _DECODER_FILTER_2_DEPTH);

  // Pass through ReLU (d_in and d_out swapped)
  cpu_relu_backward(
      _out_decoder_bias_2.get(), d_in, d_out, n, width / 2, _DECODER_FILTER_2_DEPTH);

  // Second conv2D layer
  cpu_bias_grad(d_out, d_in, n, width / 2, _DECODER_FILTER_2_DEPTH);
  cpu_update_weight(
      _decoder_bias_2.get(), d_in, _DECODER_FILTER_2_DEPTH, learning_rate);
  cpu_conv2D_grad(_out_upsampling_1.get(),
                  d_out,
                  d_filter,
                  n,
                  width / 2,
                  _DECODER_FILTER_1_DEPTH,
                  _DECODER_FILTER_2_DEPTH);
  cpu_conv2D(d_out,
             _decoder_filter_2.get(),
             d_in,
             n,
             width / 2,
             _DECODER_FILTER_1_DEPTH,
             _DECODER_FILTER_2_DEPTH);
  swap(d_out, d_in);
  cpu_update_weight(
      _decoder_filter_2.get(), d_filter, _DECODER_FILTER_2_SIZE, learning_rate);

  // Upsampling (dim: n * w/4 * w/4 * 128)
  cpu_upsampling_backward(d_out, d_in, n, width / 4, _DECODER_FILTER_1_DEPTH);

  // ReLU
  cpu_relu_backward(
      _out_decoder_bias_1.get(), d_in, d_out, n, width / 4, _DECODER_FILTER_1_DEPTH);

  // Third Conv2D
  cpu_bias_grad(d_out, d_in, n, width / 4, _DECODER_FILTER_1_DEPTH);
  cpu_update_weight(
      _decoder_bias_1.get(), d_in, _DECODER_FILTER_1_DEPTH, learning_rate);
  cpu_conv2D_grad(_out_max_pooling_2.get(),
                  d_out,
                  d_filter,
                  n,
                  width / 4,
                  _ENCODER_FILTER_2_DEPTH,
                  _DECODER_FILTER_1_DEPTH);
  cpu_conv2D(d_out,
             _decoder_filter_1.get(),
             d_in,
             n,
             width / 4,
             _ENCODER_FILTER_2_DEPTH,
             _DECODER_FILTER_1_DEPTH);
  swap(d_out, d_in);
  cpu_update_weight(
      _decoder_filter_1.get(), d_filter, _DECODER_FILTER_1_SIZE, learning_rate);

  // Max pooling backwards (dim: n * w/2 * w/2 * 128)
  cpu_max_pooling_backward(
      _out_encoder_relu_2.get(), d_out, d_in, n, width / 2, _ENCODER_FILTER_2_DEPTH);

  cpu_relu_backward(
      _out_encoder_bias_2.get(), d_in, d_out, n, width / 2, _ENCODER_FILTER_2_DEPTH);

  // Forth conv2D
  cpu_bias_grad(d_out, d_in, n, width / 2, _ENCODER_FILTER_2_DEPTH);
  cpu_update_weight(
      _encoder_bias_2.get(), d_in, _ENCODER_FILTER_2_DEPTH, learning_rate);
  cpu_conv2D_grad(_out_max_pooling_1.get(),
                  d_out,
                  d_filter,
                  n,
                  width / 2,
                  _ENCODER_FILTER_1_DEPTH,
                  _ENCODER_FILTER_2_DEPTH);
  cpu_conv2D(d_out,
             _encoder_filter_2.get(),
             d_in,
             n,
             width / 2,
             _ENCODER_FILTER_1_DEPTH,
             _ENCODER_FILTER_2_DEPTH);
  swap(d_out, d_in);
  cpu_update_weight(
      _encoder_filter_2.get(), d_filter, _ENCODER_FILTER_2_SIZE, learning_rate);

  cpu_max_pooling_backward(
      _out_encoder_bias_1.get(), d_out, d_in, n, width, _ENCODER_FILTER_1_DEPTH);
  cpu_relu_backward(
      _out_encoder_bias_1.get(), d_in, d_out, n, width, _ENCODER_FILTER_1_DEPTH);

  // Fifth conv2D
  cpu_bias_grad(d_out, d_in, n, width, _ENCODER_FILTER_1_DEPTH);
  cpu_update_weight(
      _encoder_bias_1.get(), d_in, _ENCODER_FILTER_1_DEPTH, learning_rate);
  cpu_conv2D_grad(
      batch.get_data(), d_out, d_filter, n, width, depth, _ENCODER_FILTER_1_DEPTH);
  cpu_conv2D(
      d_out, _encoder_filter_1.get(), d_in, n, width, depth, _ENCODER_FILTER_1_DEPTH);
  swap(d_out, d_in);
  cpu_update_weight(
      _encoder_filter_1.get(), d_filter, _ENCODER_FILTER_1_SIZE, learning_rate);
}

void Cpu_Autoencoder::fit(const Dataset &dataset,
                          int            n_epoch,
                          int            batch_size,
                          float          learning_rate,
                          bool           verbose,
                          int            checkpoint) {
  // Create minibatches
  vector<Dataset> batches = create_minibatches(dataset, batch_size);

  // Allocate memory for training
  _allocate_output_mem(batch_size, dataset.width);

  puts("=======================TRAINING START=======================");
  for (int epoch = 1; epoch <= n_epoch; ++epoch) {
    printf("Epoch %d:\n", epoch);

    float total_loss = 0;
    for (const Dataset &batch : batches) {
      total_loss += _fit_batch(batch, learning_rate) * batch.n;
      printf("Loss: %.4f\n", total_loss);
    }
  }
  puts("========================TRAINING END========================");

  // Deallocate memory to remove unused memory
  _deallocate_output_mem();
}

Dataset Cpu_Autoencoder::encode(const Dataset &dataset) const {
  int n     = dataset.n;
  int width = dataset.width;
  // Placeholder, alternating
  unique_ptr<float[]> a = make_unique<float[]>(n * width * width * MAX_FILTER_DEPTH);
  unique_ptr<float[]> b = make_unique<float[]>(n * width * width * MAX_FILTER_DEPTH);

  // First conv2D
  cpu_conv2D(dataset.get_data(),
             _encoder_filter_1.get(),
             a.get(),
             n,
             width,
             dataset.depth,
             _ENCODER_FILTER_1_DEPTH);
  // Add bias
  cpu_add_bias(
      a.get(), _encoder_bias_1.get(), b.get(), n, width, _ENCODER_FILTER_1_DEPTH);

  // ReLU
  cpu_relu(b.get(), a.get(), n, width, _ENCODER_FILTER_1_DEPTH);

  // Max pooling
  cpu_max_pooling(a.get(), b.get(), n, width, _ENCODER_FILTER_1_DEPTH);

  // Second conv2D
  cpu_conv2D(b.get(),
             _encoder_filter_2.get(),
             a.get(),
             n,
             width / 2,
             _ENCODER_FILTER_1_DEPTH,
             _ENCODER_FILTER_2_DEPTH);
  cpu_add_bias(
      a.get(), _encoder_bias_2.get(), b.get(), n, width / 2, _ENCODER_FILTER_2_DEPTH);

  // Second ReLU
  cpu_relu(b.get(), a.get(), n, width / 2, _ENCODER_FILTER_2_DEPTH);

  // Second max pooling
  cpu_max_pooling(a.get(), b.get(), n, width / 2, _ENCODER_FILTER_2_DEPTH);

  // Copy the result
  Dataset res(b, n, width / 4, _ENCODER_FILTER_2_DEPTH);
  memcpy(res.get_labels(), dataset.get_labels(), n * sizeof(int));
  return res;
}

Dataset Cpu_Autoencoder::decode(const Dataset &dataset) const {
  int n     = dataset.n;
  int width = dataset.width;
  // Placeholder, alternating
  unique_ptr<float[]> a = make_unique<float[]>(n * width * width * MAX_FILTER_DEPTH);
  unique_ptr<float[]> b = make_unique<float[]>(n * width * width * MAX_FILTER_DEPTH);

  // First conv2D
  cpu_conv2D(dataset.get_data(),
             _decoder_filter_1.get(),
             a.get(),
             n,
             width,
             dataset.depth,
             _DECODER_FILTER_1_DEPTH);
  // Add bias
  cpu_add_bias(
      a.get(), _decoder_bias_1.get(), b.get(), n, width, _DECODER_FILTER_1_DEPTH);

  // ReLU
  cpu_relu(b.get(), a.get(), n, width, _DECODER_FILTER_1_DEPTH);

  // Upsampling
  cpu_upsampling(a.get(), b.get(), n, width, _DECODER_FILTER_1_DEPTH);

  // Second conv2D
  cpu_conv2D(b.get(),
             _decoder_filter_2.get(),
             a.get(),
             n,
             width * 2,
             _DECODER_FILTER_1_DEPTH,
             _DECODER_FILTER_2_DEPTH);
  cpu_add_bias(
      a.get(), _decoder_bias_2.get(), b.get(), n, width * 2, _DECODER_FILTER_2_DEPTH);

  // Second ReLU
  cpu_relu(b.get(), a.get(), n, width * 2, _DECODER_FILTER_2_DEPTH);

  // Second upsampling
  cpu_upsampling(a.get(), b.get(), n, width * 2, _DECODER_FILTER_2_DEPTH);

  // Third conv2D
  cpu_conv2D(b.get(),
             _decoder_filter_3.get(),
             a.get(),
             n,
             width * 4,
             _DECODER_FILTER_2_DEPTH,
             _DECODER_FILTER_3_DEPTH);
  cpu_add_bias(
      a.get(), _decoder_bias_3.get(), b.get(), n, width * 4, _DECODER_FILTER_3_DEPTH);

  // Copy the result
  Dataset res(b, n, width * 4, _DECODER_FILTER_3_DEPTH);
  memcpy(res.get_labels(), dataset.get_labels(), n * sizeof(int));
  return res;
}