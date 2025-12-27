#include "optimized2_autoencoder.h"

/**
 * @brief Generate a random array with elements using Kaiming initialization
 *
 * @param arr The array
 * @param n The number of elements
 */
void generate_array(float *arr, float *tmp, int n, mt19937 &rng) {
  normal_distribution<float> d(0.0f, sqrt(2.0f / n));
  for (int i = 0; i < n; ++i)
    tmp[i] = d(rng);
  CUDA_CHECK(cudaMemcpy(arr, tmp, n * sizeof(float), cudaMemcpyHostToDevice));
}

/**
 * @brief Read data from a buffer
 *
 * @param buffer The buffer
 * @param data The data
 * @param size Number of bytes to read
 */
void read_data(ifstream &buffer, float *data, char *tmp, int size) {
  buffer.read(tmp, size);
  CUDA_CHECK(cudaMemcpy(data, tmp, size, cudaMemcpyHostToDevice));
}

/**
 * @brief Write data to a buffer
 *
 * @param buffer The buffer
 * @param data The data
 * @param size Number of bytes to write
 */
void write_data(ostream &buffer, float *data, char *tmp, int size) {
  CUDA_CHECK(cudaMemcpy(tmp, data, size, cudaMemcpyDeviceToHost));
  buffer.write(tmp, size);
}

Optimized2_Autoencoder::Optimized2_Autoencoder() {
  _allocate_mem();
  mt19937 rng(time(nullptr));

  static constexpr int FILTER_SIZES[]  = { ENCODER_FILTER_1_SIZE,
                                           ENCODER_FILTER_2_SIZE,
                                           DECODER_FILTER_1_SIZE,
                                           DECODER_FILTER_2_SIZE,
                                           DECODER_FILTER_3_SIZE };
  constexpr int        MAX_FILTER_SIZE = *max_element(FILTER_SIZES, FILTER_SIZES + 5);

  float *tmp;
  CUDA_CHECK(cudaMallocHost(&tmp, MAX_FILTER_SIZE * sizeof(float)));

  generate_array(_encoder_filter_1, tmp, ENCODER_FILTER_1_SIZE, rng);
  CUDA_CHECK(cudaMemset(_encoder_bias_1, 0, ENCODER_FILTER_1_DEPTH * sizeof(float)));

  generate_array(_encoder_filter_2, tmp, ENCODER_FILTER_1_SIZE, rng);
  CUDA_CHECK(cudaMemset(_encoder_bias_2, 0, ENCODER_FILTER_2_DEPTH * sizeof(float)));

  generate_array(_decoder_filter_1, tmp, ENCODER_FILTER_1_SIZE, rng);
  CUDA_CHECK(cudaMemset(_decoder_bias_1, 0, DECODER_FILTER_1_DEPTH * sizeof(float)));

  generate_array(_decoder_filter_2, tmp, ENCODER_FILTER_1_SIZE, rng);
  CUDA_CHECK(cudaMemset(_decoder_bias_2, 0, DECODER_FILTER_2_DEPTH * sizeof(float)));

  generate_array(_decoder_filter_3, tmp, ENCODER_FILTER_1_SIZE, rng);
  CUDA_CHECK(cudaMemset(_decoder_bias_3, 0, DECODER_FILTER_3_DEPTH * sizeof(float)));

  CUDA_CHECK(cudaFreeHost(tmp));
}

Optimized2_Autoencoder::Optimized2_Autoencoder(const char *filename) {
  _allocate_mem();

  // Read from file
  ifstream buffer(filename, ios::in | ios::binary);
  if (!buffer.is_open()) {
    printf("Unable to open the file %s, using random initialization.\n", filename);
    return;
  }

  static constexpr int FILTER_SIZES[]  = { ENCODER_FILTER_1_SIZE,
                                           ENCODER_FILTER_2_SIZE,
                                           DECODER_FILTER_1_SIZE,
                                           DECODER_FILTER_2_SIZE,
                                           DECODER_FILTER_3_SIZE };
  constexpr int        MAX_FILTER_SIZE = *max_element(FILTER_SIZES, FILTER_SIZES + 5);

  char *tmp;
  CUDA_CHECK(cudaMallocHost(&tmp, MAX_FILTER_SIZE * sizeof(float)));

  // Read first encoder conv2D layer
  read_data(buffer, _encoder_filter_1, tmp, ENCODER_FILTER_1_SIZE * sizeof(float));
  read_data(buffer, _encoder_bias_1, tmp, ENCODER_FILTER_1_DEPTH * sizeof(float));

  // Read second encoder conv2D layer
  read_data(buffer, _encoder_filter_2, tmp, ENCODER_FILTER_2_SIZE * sizeof(float));
  read_data(buffer, _encoder_bias_2, tmp, ENCODER_FILTER_2_DEPTH * sizeof(float));

  // Read first decoder conv2D layer
  read_data(buffer, _decoder_filter_1, tmp, DECODER_FILTER_1_SIZE * sizeof(float));
  read_data(buffer, _decoder_bias_1, tmp, DECODER_FILTER_1_DEPTH * sizeof(float));

  // Read second decoder conv2D layer
  read_data(buffer, _decoder_filter_2, tmp, DECODER_FILTER_2_SIZE * sizeof(float));
  read_data(buffer, _decoder_bias_2, tmp, DECODER_FILTER_2_DEPTH * sizeof(float));

  // Read third encoder conv2D layer
  read_data(buffer, _decoder_filter_3, tmp, DECODER_FILTER_3_SIZE * sizeof(float));
  read_data(buffer, _decoder_bias_3, tmp, DECODER_FILTER_3_DEPTH * sizeof(float));

  CUDA_CHECK(cudaFreeHost(tmp));
}

Optimized2_Autoencoder::~Optimized2_Autoencoder() {
  CUDA_CHECK(cudaFree(_encoder_filter_1));
  CUDA_CHECK(cudaFree(_encoder_bias_1));

  CUDA_CHECK(cudaFree(_encoder_filter_2));
  CUDA_CHECK(cudaFree(_encoder_bias_2));

  CUDA_CHECK(cudaFree(_decoder_filter_1));
  CUDA_CHECK(cudaFree(_decoder_bias_1));

  CUDA_CHECK(cudaFree(_decoder_filter_2));
  CUDA_CHECK(cudaFree(_decoder_bias_2));

  CUDA_CHECK(cudaFree(_decoder_filter_3));
  CUDA_CHECK(cudaFree(_decoder_bias_3));
}

void Optimized2_Autoencoder::_allocate_mem() {
  CUDA_CHECK(cudaMalloc(&_encoder_filter_1, ENCODER_FILTER_1_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_encoder_bias_1, ENCODER_FILTER_1_DEPTH * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&_encoder_filter_2, ENCODER_FILTER_2_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_encoder_bias_2, ENCODER_FILTER_2_DEPTH * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&_decoder_filter_1, DECODER_FILTER_1_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_decoder_bias_1, DECODER_FILTER_1_DEPTH * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&_decoder_filter_2, DECODER_FILTER_2_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_decoder_bias_2, DECODER_FILTER_2_DEPTH * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&_decoder_filter_3, DECODER_FILTER_3_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_decoder_bias_3, DECODER_FILTER_3_DEPTH * sizeof(float)));
}

void Optimized2_Autoencoder::_forward_pass(
    float *data, int n, int width, int height, int depth) {
  int size = n * width * height * depth;

  CUDA_CHECK(
      cudaMemcpy(_batch_data, data, size * sizeof(float), cudaMemcpyHostToDevice));

  // First conv2D layer
  optimized2_full_filter(_batch_data,
                         _encoder_filter_1,
                         _encoder_bias_1,
                         _in_encoder_relu_1,
                         _out_encoder_filter_1,
                         n,
                         width,
                         height,
                         depth,
                         ENCODER_FILTER_1_DEPTH,
                         _block_size_3D_1);

  // First max pooling layer
  optimized2_max_pooling(_out_encoder_filter_1,
                         _out_max_pooling_1,
                         n,
                         width,
                         height,
                         ENCODER_FILTER_1_DEPTH,
                         _block_size_3D_2);

  // Dim: n * w/2 * w/2 * 256
  // Second conv2D layer
  optimized2_full_filter(_out_max_pooling_1,
                         _encoder_filter_2,
                         _encoder_bias_2,
                         _in_encoder_relu_2,
                         _out_encoder_filter_2,
                         n,
                         width / 2,
                         height / 2,
                         ENCODER_FILTER_1_DEPTH,
                         ENCODER_FILTER_2_DEPTH,
                         _block_size_3D_2);

  // Second max pooling layer
  optimized2_max_pooling(_out_encoder_filter_2,
                         _out_max_pooling_2,
                         n,
                         width / 2,
                         height / 2,
                         ENCODER_FILTER_2_DEPTH,
                         _block_size_3D_3);

  // First conv2D layer
  optimized2_full_filter(_out_max_pooling_2,
                         _decoder_filter_1,
                         _decoder_bias_1,
                         _in_decoder_relu_1,
                         _out_decoder_filter_1,
                         n,
                         width / 4,
                         height / 4,
                         ENCODER_FILTER_2_DEPTH,
                         DECODER_FILTER_1_DEPTH,
                         _block_size_3D_3);

  // First upsampling layer
  optimized2_upsampling(_out_decoder_filter_1,
                        _out_upsampling_1,
                        n,
                        width / 4,
                        height / 4,
                        DECODER_FILTER_1_DEPTH,
                        _block_size_3D_2);

  // Dim: n * 2w * 2w * 256
  // Second conv2D layer
  optimized2_full_filter(_out_upsampling_1,
                         _decoder_filter_2,
                         _decoder_bias_2,
                         _in_decoder_relu_2,
                         _out_decoder_filter_2,
                         n,
                         width / 2,
                         height / 2,
                         DECODER_FILTER_1_DEPTH,
                         DECODER_FILTER_2_DEPTH,
                         _block_size_3D_2);

  // Second upsampling layer
  optimized2_upsampling(_out_decoder_filter_2,
                        _out_upsampling_2,
                        n,
                        width / 2,
                        height / 2,
                        DECODER_FILTER_2_DEPTH,
                        _block_size_3D_1);

  // Dim: n * 4w * 4w * 256
  // Third conv2D layer
  optimized2_full_filter(_out_upsampling_2,
                         _decoder_filter_3,
                         _decoder_bias_3,
                         _in_decoder_relu_3,
                         _out_decoder_filter_3,
                         n,
                         width,
                         height,
                         DECODER_FILTER_2_DEPTH,
                         DECODER_FILTER_3_DEPTH,
                         _block_size_3D_1);

  CUDA_CHECK(cudaMemcpy(_res_data,
                        _out_decoder_filter_3,
                        size * sizeof(float),
                        cudaMemcpyDeviceToDevice));
}

void Optimized2_Autoencoder::_allocate_output_mem(int n, int width, int height) {
  int n_pixel = n * width * height;

  CUDA_CHECK(cudaMalloc(&_out_encoder_filter_1,
                        n_pixel * ENCODER_FILTER_1_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_in_encoder_relu_1,
                        n_pixel * ENCODER_FILTER_1_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_out_max_pooling_1,
                        n_pixel * ENCODER_FILTER_1_DEPTH * sizeof(float) / 4));

  CUDA_CHECK(cudaMalloc(&_out_encoder_filter_2,
                        n_pixel * ENCODER_FILTER_2_DEPTH * sizeof(float) / 4));
  CUDA_CHECK(cudaMalloc(&_in_encoder_relu_2,
                        n_pixel * ENCODER_FILTER_2_DEPTH * sizeof(float) / 4));
  CUDA_CHECK(cudaMalloc(&_out_max_pooling_2,
                        n_pixel * ENCODER_FILTER_2_DEPTH * sizeof(float) / 16));

  CUDA_CHECK(cudaMalloc(&_out_decoder_filter_1,
                        n_pixel * DECODER_FILTER_1_DEPTH * sizeof(float) / 16));
  CUDA_CHECK(cudaMalloc(&_in_decoder_relu_1,
                        n_pixel * DECODER_FILTER_1_DEPTH * sizeof(float) / 16));
  CUDA_CHECK(cudaMalloc(&_out_upsampling_1,
                        n_pixel * DECODER_FILTER_1_DEPTH * sizeof(float) / 4));

  CUDA_CHECK(cudaMalloc(&_out_decoder_filter_2,
                        n_pixel * DECODER_FILTER_2_DEPTH * sizeof(float) / 4));
  CUDA_CHECK(cudaMalloc(&_in_decoder_relu_2,
                        n_pixel * DECODER_FILTER_2_DEPTH * sizeof(float) / 4));
  CUDA_CHECK(
      cudaMalloc(&_out_upsampling_2, n_pixel * DECODER_FILTER_2_DEPTH * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&_out_decoder_filter_3,
                        n_pixel * DECODER_FILTER_3_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_in_decoder_relu_3,
                        n_pixel * DECODER_FILTER_3_DEPTH * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&_batch_data, n_pixel * IMAGE_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_res_data, n_pixel * IMAGE_DEPTH * sizeof(float)));

  static constexpr int FILTER_SIZES[]  = { ENCODER_FILTER_1_SIZE,
                                           ENCODER_FILTER_2_SIZE,
                                           DECODER_FILTER_1_SIZE,
                                           DECODER_FILTER_2_SIZE,
                                           DECODER_FILTER_3_SIZE };
  constexpr int        MAX_FILTER_SIZE = *max_element(FILTER_SIZES, FILTER_SIZES + 5);

  CUDA_CHECK(cudaMalloc(&_d_in, n_pixel * MAX_FILTER_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_d_out, n_pixel * MAX_FILTER_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_d_filter, MAX_FILTER_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_d_bias, MAX_FILTER_DEPTH * sizeof(float)));
}

void Optimized2_Autoencoder::_deallocate_output_mem() {
  CUDA_CHECK(cudaFree(_out_encoder_filter_1));
  CUDA_CHECK(cudaFree(_in_encoder_relu_1));
  CUDA_CHECK(cudaFree(_out_max_pooling_1));

  CUDA_CHECK(cudaFree(_out_encoder_filter_2));
  CUDA_CHECK(cudaFree(_in_encoder_relu_2));
  CUDA_CHECK(cudaFree(_out_max_pooling_2));

  CUDA_CHECK(cudaFree(_out_decoder_filter_1));
  CUDA_CHECK(cudaFree(_in_decoder_relu_1));
  CUDA_CHECK(cudaFree(_out_upsampling_1));

  CUDA_CHECK(cudaFree(_out_decoder_filter_2));
  CUDA_CHECK(cudaFree(_in_decoder_relu_2));
  CUDA_CHECK(cudaFree(_out_upsampling_2));

  CUDA_CHECK(cudaFree(_out_decoder_filter_3));
  CUDA_CHECK(cudaFree(_in_decoder_relu_3));

  CUDA_CHECK(cudaFree(_batch_data));
  CUDA_CHECK(cudaFree(_res_data));

  CUDA_CHECK(cudaFree(_d_in));
  CUDA_CHECK(cudaFree(_d_out));
  CUDA_CHECK(cudaFree(_d_filter));
  CUDA_CHECK(cudaFree(_d_bias));
}

float Optimized2_Autoencoder::_fit_batch(
    float *data, int n, int width, int height, int depth, float learning_rate) {
  // Get the result after autoencoding
  float *d_in = _d_in, *d_out = _d_out;

  _forward_pass(data, n, width, height, depth);

  // Calculate loss before backprop
  float loss = optimized2_mse_loss(
      _batch_data, _res_data, n, width, height, depth, _block_size_1D);

  // Get loss gradient
  optimized2_mse_grad(
      _batch_data, _res_data, d_in, n, width, height, depth, _block_size_1D);

  optimized2_relu_backward(
      _in_decoder_relu_3, d_in, d_out, n, width, height, depth, _block_size_1D);

  optimized2_full_filter_grad(_out_upsampling_2,
                              d_out,
                              _d_bias,
                              _d_filter,
                              n,
                              width,
                              height,
                              DECODER_FILTER_2_DEPTH,
                              DECODER_FILTER_3_DEPTH,
                              _block_size_3D_1);

  optimized2_conv2D_backward(d_out,
                             _decoder_filter_3,
                             d_in,
                             n,
                             width,
                             height,
                             DECODER_FILTER_2_DEPTH,
                             DECODER_FILTER_3_DEPTH,
                             _block_size_3D_1);
  swap(d_in, d_out);

  // Update weight
  optimized2_update_weight(_decoder_filter_3,
                           _d_filter,
                           DECODER_FILTER_3_SIZE,
                           learning_rate,
                           _block_size_1D);

  optimized2_update_weight(
      _decoder_bias_3, _d_bias, DECODER_FILTER_3_DEPTH, learning_rate, _block_size_1D);

  // Pass through upsampling (dim: n * w/2 * w/2 * 256)
  optimized2_upsampling_backward(
      d_out, d_in, n, width / 2, height / 2, DECODER_FILTER_2_DEPTH, _block_size_3D_2);

  // Pass through ReLU (d_in and d_out swapped)
  optimized2_relu_backward(_in_decoder_relu_2,
                           d_in,
                           d_out,
                           n,
                           width / 2,
                           height / 2,
                           DECODER_FILTER_2_DEPTH,
                           _block_size_1D);

  optimized2_full_filter_grad(_out_upsampling_1,
                              d_out,
                              _d_bias,
                              _d_filter,
                              n,
                              width / 2,
                              height / 2,
                              DECODER_FILTER_1_DEPTH,
                              DECODER_FILTER_2_DEPTH,
                              _block_size_3D_2);

  optimized2_conv2D_backward(d_out,
                             _decoder_filter_2,
                             d_in,
                             n,
                             width / 2,
                             height / 2,
                             DECODER_FILTER_1_DEPTH,
                             DECODER_FILTER_2_DEPTH,
                             _block_size_3D_2);
  swap(d_in, d_out);

  // Update weight
  optimized2_update_weight(_decoder_filter_2,
                           _d_filter,
                           DECODER_FILTER_2_SIZE,
                           learning_rate,
                           _block_size_1D);

  optimized2_update_weight(
      _decoder_bias_2, _d_bias, DECODER_FILTER_2_DEPTH, learning_rate, _block_size_1D);

  // Upsampling (dim: n * w/4 * w/4 * 128)
  optimized2_upsampling_backward(
      d_out, d_in, n, width / 4, height / 4, DECODER_FILTER_1_DEPTH, _block_size_3D_3);

  // ReLU
  optimized2_relu_backward(_in_decoder_relu_1,
                           d_in,
                           d_out,
                           n,
                           width / 4,
                           height / 4,
                           DECODER_FILTER_1_DEPTH,
                           _block_size_1D);

  optimized2_full_filter_grad(_out_max_pooling_2,
                              d_out,
                              _d_bias,
                              _d_filter,
                              n,
                              width / 4,
                              height / 4,
                              ENCODER_FILTER_2_DEPTH,
                              DECODER_FILTER_1_DEPTH,
                              _block_size_3D_3);

  optimized2_conv2D_backward(d_out,
                             _decoder_filter_1,
                             d_in,
                             n,
                             width / 4,
                             height / 4,
                             ENCODER_FILTER_2_DEPTH,
                             DECODER_FILTER_1_DEPTH,
                             _block_size_3D_3);
  swap(d_in, d_out);

  // Update weight
  optimized2_update_weight(_decoder_filter_1,
                           _d_filter,
                           DECODER_FILTER_1_SIZE,
                           learning_rate,
                           _block_size_1D);

  optimized2_update_weight(
      _decoder_bias_1, _d_bias, DECODER_FILTER_1_DEPTH, learning_rate, _block_size_1D);

  // Max pooling backwards (dim: n * w/2 * w/2 * 128)
  optimized2_max_pooling_backward(_out_encoder_filter_2,
                                  d_out,
                                  d_in,
                                  n,
                                  width / 2,
                                  height / 2,
                                  ENCODER_FILTER_2_DEPTH,
                                  _block_size_3D_2);

  optimized2_relu_backward(_in_encoder_relu_2,
                           d_in,
                           d_out,
                           n,
                           width / 2,
                           height / 2,
                           ENCODER_FILTER_2_DEPTH,
                           _block_size_1D);

  optimized2_full_filter_grad(_out_max_pooling_1,
                              d_out,
                              _d_bias,
                              _d_filter,
                              n,
                              width / 2,
                              height / 2,
                              ENCODER_FILTER_1_DEPTH,
                              ENCODER_FILTER_2_DEPTH,
                              _block_size_3D_2);

  optimized2_conv2D_backward(d_out,
                             _encoder_filter_2,
                             d_in,
                             n,
                             width / 2,
                             height / 2,
                             ENCODER_FILTER_1_DEPTH,
                             ENCODER_FILTER_2_DEPTH,
                             _block_size_3D_2);
  swap(d_in, d_out);

  // Update weight
  optimized2_update_weight(_encoder_filter_2,
                           _d_filter,
                           ENCODER_FILTER_2_SIZE,
                           learning_rate,
                           _block_size_1D);

  optimized2_update_weight(
      _encoder_bias_2, _d_bias, ENCODER_FILTER_2_DEPTH, learning_rate, _block_size_1D);

  optimized2_max_pooling_backward(_out_encoder_filter_1,
                                  d_out,
                                  d_in,
                                  n,
                                  width,
                                  height,
                                  ENCODER_FILTER_1_DEPTH,
                                  _block_size_3D_1);

  optimized2_relu_backward(_in_encoder_relu_1,
                           d_in,
                           d_out,
                           n,
                           width,
                           height,
                           ENCODER_FILTER_1_DEPTH,
                           _block_size_3D_1);

  optimized2_full_filter_grad(_batch_data,
                              d_out,
                              _d_bias,
                              _d_filter,
                              n,
                              width,
                              height,
                              IMAGE_DEPTH,
                              ENCODER_FILTER_1_DEPTH,
                              _block_size_3D_1);

  // Update weight
  optimized2_update_weight(_encoder_filter_1,
                           _d_filter,
                           ENCODER_FILTER_1_SIZE,
                           learning_rate,
                           _block_size_1D);

  optimized2_update_weight(
      _encoder_bias_1, _d_bias, ENCODER_FILTER_1_DEPTH, learning_rate, _block_size_1D);

  return loss;
}

void Optimized2_Autoencoder::fit(const Optimized_Dataset &dataset,
                                 int                      n_epoch,
                                 int                      batch_size,
                                 float                    learning_rate,
                                 int                      checkpoint,
                                 const char              *output_dir) {
  // Create minibatches
  int n          = dataset.n;
  int width      = dataset.width;
  int height     = dataset.height;
  int depth      = dataset.depth;
  int image_size = width * height * depth;
  int n_batch    = (dataset.n - 1) / batch_size + 1;

  // Allocate memory for training
  _allocate_output_mem(batch_size, dataset.width, dataset.height);

  std::filesystem::create_directories(output_dir);

  Timer timer;
  float total_time = 0;

  printf(
      "Training Optimized Autoencoder (2nd version) for %d epochs with batch size %d "
      "and learning rate "
      "%.4f\n",
      n_epoch,
      batch_size,
      learning_rate);

  puts("=======================TRAINING START=======================");
  for (int epoch = 1; epoch <= n_epoch; ++epoch) {
    printf("Epoch %d/%d:\n", epoch, n_epoch);
    Progress_Bar bar(n_batch, "Batch");
    bar.update();

    int   total      = 0;
    float total_loss = 0;
    float epoch_time = 0;
    for (int i = 0; i < n_batch; ++i) {
      int offset          = i * batch_size;
      int cur_batch_size  = min(batch_size, n - offset);
      total              += cur_batch_size;

      timer.start();
      float batch_loss = _fit_batch(dataset.data + offset * image_size,
                                    cur_batch_size,
                                    width,
                                    height,
                                    depth,
                                    learning_rate);
      timer.stop();

      total_loss += batch_loss * cur_batch_size;
      epoch_time += timer.get();
      bar.update();

      printf(" - Loss = %.4f - Time = %s",
             total_loss / total,
             format_time(epoch_time).c_str());
      fflush(stdout);
    }

    total_time += epoch_time;
    puts("\n");

    // Save at checkpoints
    if (checkpoint > 0 && epoch % checkpoint == 0) {
      stringstream builder;
      builder << output_dir << '/' << "Optimized2_autoencoder_" << epoch << ".bin";
      save_parameters(builder.str().c_str());
    }
  }

  puts("========================TRAINING END========================");

  // Deallocate memory to remove unused memory
  _deallocate_output_mem();

  // Save models param
  stringstream builder;
  builder << output_dir << '/' << "Optimized2_autoencoder.bin";
  save_parameters(builder.str().c_str());

  printf("\nTotal time: %s (ms), Loss: %.4f\n",
         format_time(total_time).c_str(),
         eval(dataset));
}

Optimized_Dataset
Optimized2_Autoencoder::encode(const Optimized_Dataset &dataset) const {
  // Encode by batches to use less memory
  int width = dataset.width, height = dataset.height, depth = dataset.depth,
      n                  = dataset.n;
  int n_batch            = (n - 1) / ENCODE_BATCH_SIZE + 1;
  int image_size         = width * height * depth;
  int encoded_image_size = width / 4 * height / 4 * ENCODER_FILTER_2_DEPTH;

  Optimized_Dataset res(n, width / 4, height / 4, ENCODER_FILTER_2_DEPTH);

  // Placeholder, alternating
  float *a, *b, *c;
  CUDA_CHECK(cudaMalloc(
      &a, ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &b, ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &c, ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH * sizeof(float)));

  for (int i = 0; i < n_batch; ++i) {
    int in_offset      = i * ENCODE_BATCH_SIZE;
    int cur_batch_size = min(ENCODE_BATCH_SIZE, n - in_offset);

    CUDA_CHECK(cudaMemcpy(b,
                          dataset.data + in_offset * image_size,
                          cur_batch_size * image_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    optimized2_full_filter(b,
                           _encoder_filter_1,
                           _encoder_bias_1,
                           c,
                           a,
                           cur_batch_size,
                           width,
                           height,
                           depth,
                           ENCODER_FILTER_1_DEPTH,
                           _conv_block_size_1);

    // Max pooling
    optimized2_max_pooling(
        a, b, cur_batch_size, width, height, ENCODER_FILTER_1_DEPTH, _block_size_3D_2);

    optimized2_full_filter(b,
                           _encoder_filter_2,
                           _encoder_bias_2,
                           c,
                           a,
                           cur_batch_size,
                           width / 2,
                           height / 2,
                           ENCODER_FILTER_1_DEPTH,
                           ENCODER_FILTER_2_DEPTH,
                           _conv_block_size_2);

    // Second max pooling
    optimized2_max_pooling(a,
                           b,
                           cur_batch_size,
                           width / 2,
                           height / 2,
                           ENCODER_FILTER_2_DEPTH,
                           _block_size_3D_3);

    CUDA_CHECK(cudaMemcpy(res.data + in_offset * encoded_image_size,
                          b,
                          cur_batch_size * encoded_image_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  // Free memory
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));

  // Copy labels
  memcpy(res.labels, dataset.labels, n * sizeof(int));
  return res;
}

Optimized_Dataset
Optimized2_Autoencoder::decode(const Optimized_Dataset &dataset) const {
  int width = dataset.width, height = dataset.height, depth = dataset.depth,
      n                  = dataset.n;
  int n_batch            = (n - 1) / ENCODE_BATCH_SIZE + 1;
  int image_size         = width * height * depth;
  int decoded_image_size = 4 * width * 4 * height * DECODER_FILTER_3_DEPTH;

  Optimized_Dataset res(dataset.n, width * 4, height * 4, DECODER_FILTER_3_DEPTH);

  // Placeholder, alternating
  float *a, *b, *c;
  CUDA_CHECK(cudaMalloc(
      &a,
      ENCODE_BATCH_SIZE * 4 * width * 4 * height * MAX_FILTER_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &b,
      ENCODE_BATCH_SIZE * 4 * width * 4 * height * MAX_FILTER_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &c,
      ENCODE_BATCH_SIZE * 4 * width * 4 * height * MAX_FILTER_DEPTH * sizeof(float)));

  for (int i = 0; i < n_batch; ++i) {
    int in_offset      = i * ENCODE_BATCH_SIZE;
    int cur_batch_size = min(ENCODE_BATCH_SIZE, n - in_offset);

    CUDA_CHECK(cudaMemcpy(b,
                          dataset.data + in_offset * image_size,
                          cur_batch_size * image_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    // First conv2D
    optimized2_full_filter(b,
                           _decoder_filter_1,
                           _decoder_bias_1,
                           c,
                           a,
                           cur_batch_size,
                           width,
                           height,
                           depth,
                           DECODER_FILTER_1_DEPTH,
                           _conv_block_size_3);

    // Upsampling
    optimized2_upsampling(
        a, b, cur_batch_size, width, height, DECODER_FILTER_1_DEPTH, _block_size_3D_2);

    // Second conv2D
    optimized2_full_filter(b,
                           _decoder_filter_2,
                           _decoder_bias_2,
                           c,
                           a,
                           cur_batch_size,
                           width * 2,
                           height * 2,
                           DECODER_FILTER_1_DEPTH,
                           DECODER_FILTER_2_DEPTH,
                           _conv_block_size_3);

    // Upsampling
    optimized2_upsampling(a,
                          b,
                          cur_batch_size,
                          width * 2,
                          height * 2,
                          DECODER_FILTER_2_DEPTH,
                          _block_size_3D_1);

    optimized2_full_filter(b,
                           _decoder_filter_3,
                           _decoder_bias_3,
                           c,
                           a,
                           cur_batch_size,
                           width * 4,
                           height * 4,
                           DECODER_FILTER_2_DEPTH,
                           DECODER_FILTER_3_DEPTH,
                           _block_size_3D_2);

    CUDA_CHECK(cudaMemcpy(res.data + in_offset * decoded_image_size,
                          b,
                          cur_batch_size * decoded_image_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));

  // Copy the result
  memcpy(res.labels, dataset.labels, dataset.n * sizeof(int));
  return res;
}

float Optimized2_Autoencoder::eval(const Optimized_Dataset &dataset) const {
  int n = dataset.n, width = dataset.width, height = dataset.height,
      depth              = dataset.depth;
  int               size = n * width * height * depth;
  Optimized_Dataset res  = decode(encode(dataset));

  float *a, *b;
  CUDA_CHECK(cudaMalloc(&a, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b, size * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpyAsync(a, dataset.data, size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpyAsync(b, res.data, size * sizeof(float), cudaMemcpyHostToDevice));
  return optimized2_mse_loss(a, b, n, width, height, depth, _block_size_1D);
}

void Optimized2_Autoencoder::save_parameters(const char *filename) const {
  ofstream buffer(filename, ios::out | ios::binary);
  if (!buffer.is_open()) {
    printf("Unable to open the file %s.\n", filename);
    return;
  }

  static constexpr int FILTER_SIZES[]  = { ENCODER_FILTER_1_SIZE,
                                           ENCODER_FILTER_2_SIZE,
                                           DECODER_FILTER_1_SIZE,
                                           DECODER_FILTER_2_SIZE,
                                           DECODER_FILTER_3_SIZE };
  constexpr int        MAX_FILTER_SIZE = *max_element(FILTER_SIZES, FILTER_SIZES + 5);

  char *tmp;
  CUDA_CHECK(cudaMallocHost(&tmp, MAX_FILTER_SIZE * sizeof(float)));

  // Write first encoder conv2D layer
  write_data(buffer, _encoder_filter_1, tmp, ENCODER_FILTER_1_SIZE * sizeof(float));
  write_data(buffer, _encoder_bias_1, tmp, ENCODER_FILTER_1_DEPTH * sizeof(float));

  // Write second encoder conv2D layer
  write_data(buffer, _encoder_filter_2, tmp, ENCODER_FILTER_2_SIZE * sizeof(float));
  write_data(buffer, _encoder_bias_2, tmp, ENCODER_FILTER_2_DEPTH * sizeof(float));

  // Write first decoder conv2D layer
  write_data(buffer, _decoder_filter_1, tmp, DECODER_FILTER_1_SIZE * sizeof(float));
  write_data(buffer, _decoder_bias_1, tmp, DECODER_FILTER_1_DEPTH * sizeof(float));

  // Write second decoder conv2D layer
  write_data(buffer, _decoder_filter_2, tmp, DECODER_FILTER_2_SIZE * sizeof(float));
  write_data(buffer, _decoder_bias_2, tmp, DECODER_FILTER_2_DEPTH * sizeof(float));

  // Write third encoder conv2D layer
  write_data(buffer, _decoder_filter_3, tmp, DECODER_FILTER_3_SIZE * sizeof(float));
  write_data(buffer, _decoder_bias_3, tmp, DECODER_FILTER_3_DEPTH * sizeof(float));

  CUDA_CHECK(cudaFreeHost(tmp));
}