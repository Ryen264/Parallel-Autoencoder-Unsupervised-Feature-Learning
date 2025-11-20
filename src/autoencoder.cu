#include "autoencoder.h"
#include "constants.h"
#include <cstdlib>
#include <fstream>

using std::ifstream, std::ofstream;
using std::make_unique;

/**
 * @brief Generate a random number between 0 and 1
 *
 * @return float The generated number
 */
float generate() { return 1.0 * rand() / RAND_MAX; }

void IAutoencoder::_allocate_mem() {
  _encoder_filter_1 = make_unique<float[]>(_ENCODER_FILTER_1_SIZE);
  _encoder_bias_1   = make_unique<float[]>(_ENCODER_FILTER_1_DEPTH);
  _encoder_filter_2 = make_unique<float[]>(_ENCODER_FILTER_2_SIZE);
  _encoder_bias_2   = make_unique<float[]>(_ENCODER_FILTER_2_DEPTH);
  _decoder_filter_1 = make_unique<float[]>(_DECODER_FILTER_1_SIZE);
  _decoder_bias_1   = make_unique<float[]>(_DECODER_FILTER_1_DEPTH);
  _decoder_filter_2 = make_unique<float[]>(_DECODER_FILTER_2_SIZE);
  _decoder_bias_2   = make_unique<float[]>(_DECODER_FILTER_2_DEPTH);
  _decoder_filter_1 = make_unique<float[]>(_DECODER_FILTER_3_SIZE);
  _decoder_bias_1   = make_unique<float[]>(_DECODER_FILTER_3_DEPTH);
}

IAutoencoder::IAutoencoder() {
  _allocate_mem();

  // Random init
  srand(time(0));

  float *encoder_filter_1 = _encoder_filter_1.get();
  float *encoder_bias_1   = _encoder_bias_1.get();
  for (int i = 0; i < _ENCODER_FILTER_1_SIZE; ++i)
    encoder_filter_1[i] = generate();
  for (int i = 0; i < _ENCODER_FILTER_1_DEPTH; ++i)
    encoder_bias_1[i] = generate();

  float *encoder_filter_1 = _encoder_filter_1.get();
  float *encoder_bias_1   = _encoder_bias_1.get();
  for (int i = 0; i < _ENCODER_FILTER_1_SIZE; ++i)
    encoder_filter_1[i] = generate();
  for (int i = 0; i < _ENCODER_FILTER_1_DEPTH; ++i)
    encoder_bias_1[i] = generate();

  float *encoder_filter_2 = _encoder_filter_2.get();
  float *encoder_bias_2   = _encoder_bias_2.get();
  for (int i = 0; i < _ENCODER_FILTER_2_SIZE; ++i)
    encoder_filter_2[i] = generate();
  for (int i = 0; i < _ENCODER_FILTER_2_DEPTH; ++i)
    encoder_bias_2[i] = generate();

  float *decoder_filter_1 = _decoder_filter_1.get();
  float *decoder_bias_1   = _decoder_bias_1.get();
  for (int i = 0; i < _DECODER_FILTER_1_SIZE; ++i)
    decoder_filter_1[i] = generate();
  for (int i = 0; i < _DECODER_FILTER_1_DEPTH; ++i)
    decoder_bias_1[i] = generate();

  float *decoder_filter_2 = _decoder_filter_2.get();
  float *decoder_bias_2   = _decoder_bias_2.get();
  for (int i = 0; i < _DECODER_FILTER_2_SIZE; ++i)
    decoder_filter_2[i] = generate();
  for (int i = 0; i < _DECODER_FILTER_2_DEPTH; ++i)
    decoder_bias_2[i] = generate();

  float *decoder_filter_3 = _decoder_filter_3.get();
  float *decoder_bias_3   = _decoder_bias_3.get();
  for (int i = 0; i < _DECODER_FILTER_3_SIZE; ++i)
    decoder_filter_3[i] = generate();
  for (int i = 0; i < _DECODER_FILTER_3_DEPTH; ++i)
    decoder_bias_3[i] = generate();
}

IAutoencoder::IAutoencoder(const char *filename) {
  _allocate_mem();

  // Read from file
  ifstream buffer(filename, std::ios::in | std::ios::binary);

  // Read first encoder conv2D layer
  char *encoder_filter_1 = reinterpret_cast<char *>(_encoder_filter_1.get());
  char *encoder_bias_1   = reinterpret_cast<char *>(_encoder_bias_1.get());
  buffer.read(encoder_filter_1, _ENCODER_FILTER_1_SIZE * sizeof(float));
  buffer.read(encoder_bias_1, _ENCODER_FILTER_1_DEPTH * sizeof(float));

  // Read second encoder conv2D layer
  char *encoder_filter_2 = reinterpret_cast<char *>(_encoder_filter_2.get());
  char *encoder_bias_2   = reinterpret_cast<char *>(_encoder_bias_2.get());
  buffer.read(encoder_filter_2, _ENCODER_FILTER_2_SIZE * sizeof(float));
  buffer.read(encoder_bias_2, _ENCODER_FILTER_2_DEPTH * sizeof(float));

  // Read first decoder conv2D layer
  char *decoder_filter_1 = reinterpret_cast<char *>(_decoder_filter_1.get());
  char *decoder_bias_1   = reinterpret_cast<char *>(_decoder_bias_1.get());
  buffer.read(decoder_filter_1, _DECODER_FILTER_1_SIZE * sizeof(float));
  buffer.read(decoder_bias_1, _DECODER_FILTER_1_DEPTH * sizeof(float));

  // Read second decoder conv2D layer
  char *decoder_filter_2 = reinterpret_cast<char *>(_decoder_filter_2.get());
  char *decoder_bias_2   = reinterpret_cast<char *>(_decoder_bias_2.get());
  buffer.read(decoder_filter_2, _DECODER_FILTER_2_SIZE * sizeof(float));
  buffer.read(decoder_bias_2, _DECODER_FILTER_2_DEPTH * sizeof(float));

  // Read third encoder conv2D layer
  char *decoder_filter_3 = reinterpret_cast<char *>(_decoder_filter_3.get());
  char *decoder_bias_3   = reinterpret_cast<char *>(_decoder_bias_3.get());
  buffer.read(decoder_filter_3, _DECODER_FILTER_3_SIZE * sizeof(float));
  buffer.read(decoder_bias_3, _DECODER_FILTER_3_DEPTH * sizeof(float));
}

void IAutoencoder::_save_paramters(const char *filename) const {
  ofstream buffer(filename, std::ios::out | std::ios::binary);

  // Write first encoder conv2D layer
  const char *encoder_filter_1 =
      reinterpret_cast<const char *>(_encoder_filter_1.get());
  const char *encoder_bias_1 = reinterpret_cast<const char *>(_encoder_bias_1.get());
  buffer.write(encoder_filter_1, _ENCODER_FILTER_1_SIZE * sizeof(float));
  buffer.write(encoder_bias_1, _ENCODER_FILTER_1_DEPTH * sizeof(float));

  // Write second encoder conv2D layer
  const char *encoder_filter_2 =
      reinterpret_cast<const char *>(_encoder_filter_2.get());
  const char *encoder_bias_2 = reinterpret_cast<const char *>(_encoder_bias_2.get());
  buffer.write(encoder_filter_2, _ENCODER_FILTER_2_SIZE * sizeof(float));
  buffer.write(encoder_bias_2, _ENCODER_FILTER_2_DEPTH * sizeof(float));

  // Write first decoder conv2D layer
  const char *decoder_filter_1 =
      reinterpret_cast<const char *>(_decoder_filter_1.get());
  const char *decoder_bias_1 = reinterpret_cast<const char *>(_decoder_bias_1.get());
  buffer.write(decoder_filter_1, _DECODER_FILTER_1_SIZE * sizeof(float));
  buffer.write(decoder_bias_1, _DECODER_FILTER_1_DEPTH * sizeof(float));

  // Write second decoder conv2D layer
  const char *decoder_filter_2 =
      reinterpret_cast<const char *>(_decoder_filter_2.get());
  const char *decoder_bias_2 = reinterpret_cast<const char *>(_decoder_bias_2.get());
  buffer.write(decoder_filter_2, _DECODER_FILTER_2_SIZE * sizeof(float));
  buffer.write(decoder_bias_2, _DECODER_FILTER_2_DEPTH * sizeof(float));

  // Write third encoder conv2D layer
  const char *decoder_filter_3 =
      reinterpret_cast<const char *>(_decoder_filter_3.get());
  const char *decoder_bias_3 = reinterpret_cast<const char *>(_decoder_bias_3.get());
  buffer.write(decoder_filter_3, _DECODER_FILTER_3_SIZE * sizeof(float));
  buffer.write(decoder_bias_3, _DECODER_FILTER_3_DEPTH * sizeof(float));
}