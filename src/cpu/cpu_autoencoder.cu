#include "constants.h"
#include "cpu_autoencoder.h"
#include "cpu_layers.h"
#include <cstring>

using std::memcpy;
using std::swap;

Cpu_Autoencoder::Cpu_Autoencoder()
    : IAutoencoder() {};

Cpu_Autoencoder::Cpu_Autoencoder(const char *filename)
    : IAutoencoder(filename) {};

Images Cpu_Autoencoder::encode(const Images &images) const {
  int n     = images.n;
  int width = images.width;

  // First conv2D layer
  cpu_conv2D(images.get(),
             _encoder_filter_1.get(),
             _out_encoder_filter_1.get(),
             n,
             width,
             images.depth,
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
  Images res(n, width / 4, _ENCODER_FILTER_2_DEPTH);
  memcpy(res.get(),
         _out_max_pooling_2.get(),
         n * width * width * _ENCODER_FILTER_2_DEPTH * sizeof(float) / 16);

  return res;
}

Images Cpu_Autoencoder::decode(const Images &images) const {
  int n     = images.n;
  int width = images.width;

  // First conv2D layer
  cpu_conv2D(images.get(),
             _decoder_filter_1.get(),
             _out_decoder_filter_1.get(),
             n,
             width,
             images.depth,
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
  Images res(n, width / 4, _ENCODER_FILTER_2_DEPTH);
  memcpy(res.get(),
         _out_decoder_bias_3.get(),
         n * width * width * _DECODER_FILTER_3_DEPTH * 16 * sizeof(float));

  return res;
}

float Cpu_Autoencoder::_fit_batch(const Images &images,
                                  float        *d_out,
                                  float        *d_in,
                                  float        *d_filter,
                                  float         learning_rate) {
  // Get the result after autoencoding
  int    n     = images.n;
  int    width = images.width;
  Images res   = decode(encode(images));

  // Get loss gradient
  cpu_mse_grad(images.get(), res.get(), d_out, n, width, images.depth);

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
      images.get(), d_out, d_filter, n, width, images.depth, _ENCODER_FILTER_1_DEPTH);
  cpu_conv2D(d_out,
             _encoder_filter_1.get(),
             d_in,
             n,
             width,
             images.depth,
             _ENCODER_FILTER_1_DEPTH);
  swap(d_out, d_in);
  cpu_update_weight(
      _encoder_filter_1.get(), d_filter, _ENCODER_FILTER_1_SIZE, learning_rate);
}

void Cpu_Autoencoder::fit(const Images &images,
                          int           n_epoch,
                          int           batch_size,
                          float         learning_rate,
                          bool          verbose,
                          int           checkpoint) {
  // Create minibatches
}