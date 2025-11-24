#include <cstdlib>
#include <fstream>
#include "autoencoder.h"
#include "constants.h"

using std::ifstream, std::ofstream;
using std::make_unique;

/**
 * @brief Generate a random array with elements between 0 and 1
 *
 * @param arr The array
 * @param n The number of elements
 */
void generate_array(const std::unique_ptr<float[]> &arr, int n) {
    float *ptr = arr.get();
    for (int i = 0; i < n; ++i)
      ptr[i] = 1.0f * rand() / RAND_MAX;
}

/**
 * @brief Read data from a buffer
 *
 * @param buffer The buffer
 * @param data The data
 * @param size Number of bytes to read
 */
void read_data(std::ifstream &buffer, const std::unique_ptr<float[]> &data, int size) {
    char *ptr = reinterpret_cast<char *>(data.get());
    buffer.read(ptr, size);
}

/**
 * @brief Write data to a buffer
 *
 * @param buffer The buffer
 * @param data The data
 * @param size Number of bytes to write
 */
void write_data(std::ostream &buffer, const std::unique_ptr<float[]> &data, int size) {
    const char *ptr = reinterpret_cast<const char *>(data.get());
    buffer.write(ptr, size);
}

void IAutoencoder::_allocate_mem() {
    _encoder_filter_1 = make_unique<float[]>(_ENCODER_FILTER_1_SIZE);
    _encoder_bias_1   = make_unique<float[]>(_ENCODER_FILTER_1_DEPTH);

    _encoder_filter_2 = make_unique<float[]>(_ENCODER_FILTER_2_SIZE);
    _encoder_bias_2   = make_unique<float[]>(_ENCODER_FILTER_2_DEPTH);

    _decoder_filter_1 = make_unique<float[]>(_DECODER_FILTER_1_SIZE);
    _decoder_bias_1   = make_unique<float[]>(_DECODER_FILTER_1_DEPTH);

    _decoder_filter_2 = make_unique<float[]>(_DECODER_FILTER_2_SIZE);
    _decoder_bias_2   = make_unique<float[]>(_DECODER_FILTER_2_DEPTH);

    _decoder_filter_3 = make_unique<float[]>(_DECODER_FILTER_3_SIZE);
    _decoder_bias_3   = make_unique<float[]>(_DECODER_FILTER_3_DEPTH);
}

IAutoencoder::IAutoencoder() {
    _allocate_mem();

    // Random init
    srand(time(0));

    generate_array(_encoder_filter_1, _ENCODER_FILTER_1_SIZE);
    generate_array(_encoder_bias_1, _ENCODER_FILTER_1_DEPTH);

    generate_array(_encoder_filter_2, _ENCODER_FILTER_2_SIZE);
    generate_array(_encoder_bias_2, _ENCODER_FILTER_2_DEPTH);

    generate_array(_decoder_filter_1, _DECODER_FILTER_1_SIZE);
    generate_array(_decoder_bias_1, _DECODER_FILTER_1_DEPTH);

    generate_array(_decoder_filter_2, _DECODER_FILTER_2_SIZE);
    generate_array(_decoder_bias_2, _DECODER_FILTER_2_DEPTH);

    generate_array(_decoder_filter_2, _DECODER_FILTER_2_SIZE);
    generate_array(_decoder_bias_2, _DECODER_FILTER_2_DEPTH);
}

IAutoencoder::IAutoencoder(const char *filename) {
    _allocate_mem();

    // Read from file
    ifstream buffer(filename, std::ios::in | std::ios::binary);

    // Read first encoder conv2D layer
    read_data(buffer, _encoder_filter_1, _ENCODER_FILTER_1_SIZE * sizeof(float));
    read_data(buffer, _encoder_bias_1, _ENCODER_FILTER_1_DEPTH * sizeof(float));

    // Read second encoder conv2D layer
    read_data(buffer, _encoder_filter_2, _ENCODER_FILTER_2_SIZE * sizeof(float));
    read_data(buffer, _encoder_bias_2, _ENCODER_FILTER_2_DEPTH * sizeof(float));

    // Read first decoder conv2D layer
    read_data(buffer, _decoder_filter_1, _DECODER_FILTER_1_SIZE * sizeof(float));
    read_data(buffer, _decoder_bias_1, _DECODER_FILTER_1_DEPTH * sizeof(float));

    // Read second decoder conv2D layer
    read_data(buffer, _decoder_filter_2, _DECODER_FILTER_2_SIZE * sizeof(float));
    read_data(buffer, _decoder_bias_2, _DECODER_FILTER_2_DEPTH * sizeof(float));

    // Read third encoder conv2D layer
    read_data(buffer, _decoder_filter_3, _DECODER_FILTER_3_SIZE * sizeof(float));
    read_data(buffer, _decoder_bias_3, _DECODER_FILTER_3_DEPTH * sizeof(float));
}

void IAutoencoder::_save_paramters(const char *filename) const {
    ofstream buffer(filename, std::ios::out | std::ios::binary);

    // Write first encoder conv2D layer
    write_data(buffer, _encoder_filter_1, _ENCODER_FILTER_1_SIZE * sizeof(float));
    write_data(buffer, _encoder_bias_1, _ENCODER_FILTER_1_DEPTH * sizeof(float));

    // Write second encoder conv2D layer
    write_data(buffer, _encoder_filter_2, _ENCODER_FILTER_2_SIZE * sizeof(float));
    write_data(buffer, _encoder_bias_2, _ENCODER_FILTER_2_DEPTH * sizeof(float));

    // Write first decoder conv2D layer
    write_data(buffer, _decoder_filter_1, _DECODER_FILTER_1_SIZE * sizeof(float));
    write_data(buffer, _decoder_bias_1, _DECODER_FILTER_1_DEPTH * sizeof(float));

    // Write second decoder conv2D layer
    write_data(buffer, _decoder_filter_2, _DECODER_FILTER_2_SIZE * sizeof(float));
    write_data(buffer, _decoder_bias_2, _DECODER_FILTER_2_DEPTH * sizeof(float));

    // Write third encoder conv2D layer
    write_data(buffer, _decoder_filter_3, _DECODER_FILTER_3_SIZE * sizeof(float));
    write_data(buffer, _decoder_bias_3, _DECODER_FILTER_3_DEPTH * sizeof(float));
}