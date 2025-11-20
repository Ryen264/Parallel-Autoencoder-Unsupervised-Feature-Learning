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
  // Result holder
  int                 n     = images.n;
  int                 width = images.width;
  unique_ptr<float[]> in    = make_unique<float[]>(n * MAX_IMAGE_SIZE);
  unique_ptr<float[]> out   = make_unique<float[]>(n * MAX_IMAGE_SIZE);

  // First conv2D layer
  cpu_conv2D(images.get(),
             _encoder_filter_1.get(),
             in.get(),
             n,
             width,
             images.depth,
             _ENCODER_FILTER_1_DEPTH);
  // Dim: n * w * w * 256
  cpu_add_bias(
      in.get(), _encoder_bias_1.get(), out.get(), n, width, _ENCODER_FILTER_1_DEPTH);
  // Copy result to in
  swap(in, out);

  // ReLU layer
  cpu_relu(in.get(), out.get(), n, width, _ENCODER_FILTER_1_DEPTH);
  // Copy result to in
  swap(in, out);

  // First max pooling layer
  cpu_max_pooling(in.get(), out.get(), n, width, _ENCODER_FILTER_1_DEPTH);
  // Copy result to in
  swap(in, out);

  // Dim: n * w/2 * w/2 * 256
  // Second conv2D layer
  cpu_conv2D(in.get(),
             _encoder_filter_2.get(),
             out.get(),
             n,
             width / 2,
             _ENCODER_FILTER_1_DEPTH,
             _ENCODER_FILTER_2_DEPTH);
  // Copy result to in
  swap(in, out);
  // Dim: n * w/2 * w/2 * 128
  cpu_add_bias(in.get(),
               _encoder_bias_1.get(),
               out.get(),
               n,
               width / 2,
               _ENCODER_FILTER_2_DEPTH);
  // Copy result to in
  swap(in, out);

  // ReLU layer
  cpu_relu(in.get(), out.get(), n, width / 2, _ENCODER_FILTER_2_DEPTH);
  // Copy result to in
  swap(in, out);

  // Second max pooling layer
  cpu_max_pooling(in.get(), out.get(), n, width / 2, _ENCODER_FILTER_2_DEPTH);

  // Realloc to remove unused space
  // Dim: n * w/4 * w/4 * 128
  int total_size = n * width * width * _ENCODER_FILTER_2_DEPTH * sizeof(float) / 16;
  in             = make_unique<float[]>(total_size);
  memcpy(in.get(), out.get(), total_size);

  // Return the result (Dim: n * w/4 * w/4 * 128)
  return Images(in, n, width / 4, _ENCODER_FILTER_2_DEPTH);
}

Images Cpu_Autoencoder::decode(const Images &images) const {
  // Result holder
  int                 n     = images.n;
  int                 width = images.width;
  unique_ptr<float[]> in    = make_unique<float[]>(n * MAX_IMAGE_SIZE);
  unique_ptr<float[]> out   = make_unique<float[]>(n * MAX_IMAGE_SIZE);

  // First conv2D layer
  cpu_conv2D(images.get(),
             _decoder_filter_1.get(),
             in.get(),
             n,
             width,
             images.depth,
             _DECODER_FILTER_1_DEPTH);
  // Dim: n * w * w * 128
  cpu_add_bias(
      in.get(), _encoder_bias_1.get(), out.get(), n, width, _DECODER_FILTER_1_DEPTH);
  // Copy result to in
  swap(in, out);

  // ReLU layer
  cpu_relu(in.get(), out.get(), n, width, _DECODER_FILTER_1_DEPTH);
  // Copy result to in
  swap(in, out);

  // First upsampling layer
  cpu_upsampling(in.get(), out.get(), n, width, _DECODER_FILTER_1_DEPTH);
  // Copy result to in
  swap(in, out);

  // Dim: n * 2w * 2w * 256
  // Second conv2D layer
  cpu_conv2D(in.get(),
             _decoder_filter_2.get(),
             out.get(),
             n,
             2 * width,
             _DECODER_FILTER_1_DEPTH,
             _DECODER_FILTER_2_DEPTH);
  // Copy result to in
  swap(in, out);
  // Dim: n * 2w * 2w * 256
  cpu_add_bias(in.get(),
               _decoder_bias_2.get(),
               out.get(),
               n,
               2 * width,
               _DECODER_FILTER_2_DEPTH);
  // Copy result to in
  swap(in, out);

  // ReLU layer
  cpu_relu(in.get(), out.get(), n, 2 * width, _DECODER_FILTER_2_DEPTH);
  // Copy result to in
  swap(in, out);

  // Second upsampling layer
  cpu_upsampling(in.get(), out.get(), n, 2 * width, _DECODER_FILTER_2_DEPTH);
  // Copy result to in
  swap(in, out);

  // Dim: n * 4w * 4w * 256
  // Third conv2D layer
  cpu_conv2D(in.get(),
             _decoder_filter_3.get(),
             out.get(),
             n,
             4 * width,
             _DECODER_FILTER_2_DEPTH,
             _DECODER_FILTER_3_DEPTH);
  // Copy result to in
  swap(in, out);
  // Dim: n * 4w * 4w * 3
  cpu_add_bias(in.get(),
               _decoder_bias_3.get(),
               out.get(),
               n,
               4 * width,
               _DECODER_FILTER_3_DEPTH);

  // Realloc to remove unused space
  int total_size = n * width * width * _DECODER_FILTER_3_DEPTH * sizeof(float) * 16;
  in             = make_unique<float[]>(total_size);
  memcpy(in.get(), out.get(), total_size);

  // Return the result (Dim: n * 4w * 4w * 3)
  return Images(in, n, 4 * width, _DECODER_FILTER_3_DEPTH);
}