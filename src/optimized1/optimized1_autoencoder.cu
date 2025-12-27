#include "optimized1_autoencoder.h"
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#endif

// =================================================================================
// HELPER FUNCTIONS (STATIC TO PREVENT LINKER ERRORS)
// =================================================================================

/**
 * @brief Generate a random array using Kaiming initialization (CPU init -> GPU copy)
 */
static void generate_array(float *arr, int n, int fan_in, mt19937 &rng) {
  vector<float>              tmp(n);
  normal_distribution<float> d(0.0f, sqrt(2.0f / fan_in));
  for (int i = 0; i < n; ++i)
    tmp[i] = d(rng);
  CUDA_CHECK(cudaMemcpy(arr, tmp.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}

/**
 * @brief Read data from a buffer (Handles temp memory internally)
 */
static void read_data(ifstream &buffer, float *data, int size) {
  unique_ptr<char[]> ptr = make_unique<char[]>(size);
  buffer.read(ptr.get(), size);
  CUDA_CHECK(cudaMemcpy(data, ptr.get(), size, cudaMemcpyHostToDevice));
}

/**
 * @brief Write data to a buffer (Handles temp memory internally)
 */
static void write_data(ostream &buffer, float *data, int size) {
  unique_ptr<char[]> ptr = make_unique<char[]>(size);
  CUDA_CHECK(cudaMemcpy(ptr.get(), data, size, cudaMemcpyDeviceToHost));
  buffer.write(ptr.get(), size);
}

// =================================================================================
// CLASS IMPLEMENTATION
// =================================================================================

Optimized1_Autoencoder::Optimized1_Autoencoder() {
  _allocate_mem();
  mt19937 rng(time(nullptr));
  int     n_in = CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT;

  // Random Init
  srand(time(0));

  generate_array(_encoder_filter_1, ENCODER_FILTER_1_SIZE, n_in * IMAGE_SIZE, rng);
  CUDA_CHECK(cudaMemset(_encoder_bias_1, 0, ENCODER_FILTER_1_DEPTH * sizeof(float)));

  generate_array(
      _encoder_filter_2, ENCODER_FILTER_2_SIZE, n_in * ENCODER_FILTER_1_DEPTH, rng);
  CUDA_CHECK(cudaMemset(_encoder_bias_2, 0, ENCODER_FILTER_2_DEPTH * sizeof(float)));

  generate_array(
      _decoder_filter_1, DECODER_FILTER_1_SIZE, n_in * ENCODER_FILTER_2_DEPTH, rng);
  CUDA_CHECK(cudaMemset(_decoder_bias_1, 0, DECODER_FILTER_1_DEPTH * sizeof(float)));

  generate_array(
      _decoder_filter_2, DECODER_FILTER_2_SIZE, n_in * DECODER_FILTER_1_DEPTH, rng);
  CUDA_CHECK(cudaMemset(_decoder_bias_2, 0, DECODER_FILTER_2_DEPTH * sizeof(float)));

  generate_array(
      _decoder_filter_3, DECODER_FILTER_3_SIZE, n_in * DECODER_FILTER_2_DEPTH, rng);
  CUDA_CHECK(cudaMemset(_decoder_bias_3, 0, DECODER_FILTER_3_DEPTH * sizeof(float)));
}

Optimized1_Autoencoder::Optimized1_Autoencoder(const char *filename) {
  _allocate_mem();
  load_parameters(filename); // Reuse load logic
}

Optimized1_Autoencoder::~Optimized1_Autoencoder() {
  _deallocate_output_mem();

  if (_encoder_filter_1)
    cudaFree(_encoder_filter_1);
  if (_encoder_bias_1)
    cudaFree(_encoder_bias_1);
  if (_encoder_filter_2)
    cudaFree(_encoder_filter_2);
  if (_encoder_bias_2)
    cudaFree(_encoder_bias_2);

  if (_decoder_filter_1)
    cudaFree(_decoder_filter_1);
  if (_decoder_bias_1)
    cudaFree(_decoder_bias_1);
  if (_decoder_filter_2)
    cudaFree(_decoder_filter_2);
  if (_decoder_bias_2)
    cudaFree(_decoder_bias_2);
  if (_decoder_filter_3)
    cudaFree(_decoder_filter_3);
  if (_decoder_bias_3)
    cudaFree(_decoder_bias_3);
}

void Optimized1_Autoencoder::_allocate_mem() {
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

void Optimized1_Autoencoder::_allocate_output_mem(int n, int width, int height) {
  int n_pixel = n * width * height;

  CUDA_CHECK(cudaMalloc(&_out_encoder_filter_1,
                        n_pixel * ENCODER_FILTER_1_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_out_encoder_bias_1,
                        n_pixel * ENCODER_FILTER_1_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_out_encoder_relu_1,
                        n_pixel * ENCODER_FILTER_1_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_out_max_pooling_1,
                        n_pixel * ENCODER_FILTER_1_DEPTH * sizeof(float) / 4));

  CUDA_CHECK(cudaMalloc(&_out_encoder_filter_2,
                        n_pixel * ENCODER_FILTER_2_DEPTH * sizeof(float) / 4));
  CUDA_CHECK(cudaMalloc(&_out_encoder_bias_2,
                        n_pixel * ENCODER_FILTER_2_DEPTH * sizeof(float) / 4));
  CUDA_CHECK(cudaMalloc(&_out_encoder_relu_2,
                        n_pixel * ENCODER_FILTER_2_DEPTH * sizeof(float) / 4));
  CUDA_CHECK(cudaMalloc(&_out_max_pooling_2,
                        n_pixel * ENCODER_FILTER_2_DEPTH * sizeof(float) / 16));

  CUDA_CHECK(cudaMalloc(&_out_decoder_filter_1,
                        n_pixel * DECODER_FILTER_1_DEPTH * sizeof(float) / 16));
  CUDA_CHECK(cudaMalloc(&_out_decoder_bias_1,
                        n_pixel * DECODER_FILTER_1_DEPTH * sizeof(float) / 16));
  CUDA_CHECK(cudaMalloc(&_out_decoder_relu_1,
                        n_pixel * DECODER_FILTER_1_DEPTH * sizeof(float) / 16));
  CUDA_CHECK(cudaMalloc(&_out_upsampling_1,
                        n_pixel * DECODER_FILTER_1_DEPTH * sizeof(float) / 4));

  CUDA_CHECK(cudaMalloc(&_out_decoder_filter_2,
                        n_pixel * DECODER_FILTER_2_DEPTH * sizeof(float) / 4));
  CUDA_CHECK(cudaMalloc(&_out_decoder_bias_2,
                        n_pixel * DECODER_FILTER_2_DEPTH * sizeof(float) / 4));
  CUDA_CHECK(cudaMalloc(&_out_decoder_relu_2,
                        n_pixel * DECODER_FILTER_2_DEPTH * sizeof(float) / 4));
  CUDA_CHECK(
      cudaMalloc(&_out_upsampling_2, n_pixel * DECODER_FILTER_2_DEPTH * sizeof(float)));

  CUDA_CHECK(cudaMalloc(&_out_decoder_filter_3,
                        n_pixel * DECODER_FILTER_3_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&_out_decoder_bias_3,
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
}

void Optimized1_Autoencoder::_deallocate_output_mem() {
  if (_out_encoder_filter_1)
    cudaFree(_out_encoder_filter_1);
  if (_out_encoder_bias_1)
    cudaFree(_out_encoder_bias_1);
  if (_out_encoder_relu_1)
    cudaFree(_out_encoder_relu_1);
  if (_out_max_pooling_1)
    cudaFree(_out_max_pooling_1);

  if (_out_encoder_filter_2)
    cudaFree(_out_encoder_filter_2);
  if (_out_encoder_bias_2)
    cudaFree(_out_encoder_bias_2);
  if (_out_encoder_relu_2)
    cudaFree(_out_encoder_relu_2);
  if (_out_max_pooling_2)
    cudaFree(_out_max_pooling_2);

  if (_out_decoder_filter_1)
    cudaFree(_out_decoder_filter_1);
  if (_out_decoder_bias_1)
    cudaFree(_out_decoder_bias_1);
  if (_out_decoder_relu_1)
    cudaFree(_out_decoder_relu_1);
  if (_out_upsampling_1)
    cudaFree(_out_upsampling_1);

  if (_out_decoder_filter_2)
    cudaFree(_out_decoder_filter_2);
  if (_out_decoder_bias_2)
    cudaFree(_out_decoder_bias_2);
  if (_out_decoder_relu_2)
    cudaFree(_out_decoder_relu_2);
  if (_out_upsampling_2)
    cudaFree(_out_upsampling_2);

  if (_out_decoder_filter_3)
    cudaFree(_out_decoder_filter_3);
  if (_out_decoder_bias_3)
    cudaFree(_out_decoder_bias_3);

  if (_batch_data)
    cudaFree(_batch_data);
  if (_res_data)
    cudaFree(_res_data);
  if (_d_in)
    cudaFree(_d_in);
  if (_d_out)
    cudaFree(_d_out);
  if (_d_filter)
    cudaFree(_d_filter);
}

void Optimized1_Autoencoder::_forward_pass(
    float *data, int n, int width, int height, int depth) {
  int size = n * width * height * depth;
  CUDA_CHECK(
      cudaMemcpy(_batch_data, data, size * sizeof(float), cudaMemcpyHostToDevice));

  // First conv2D layer
  optimized1_conv2D(_batch_data,
                    _encoder_filter_1,
                    _out_encoder_filter_1,
                    n,
                    width,
                    height,
                    depth,
                    ENCODER_FILTER_1_DEPTH,
                    _block_size_3D_1);
  optimized1_add_bias(_out_encoder_filter_1,
                      _encoder_bias_1,
                      _out_encoder_bias_1,
                      n,
                      width,
                      height,
                      ENCODER_FILTER_1_DEPTH,
                      _block_size_1D);
  optimized1_relu(_out_encoder_bias_1,
                  _out_encoder_relu_1,
                  n,
                  width,
                  height,
                  ENCODER_FILTER_1_DEPTH,
                  _block_size_1D);
  optimized1_max_pooling(_out_encoder_relu_1,
                         _out_max_pooling_1,
                         n,
                         width,
                         height,
                         ENCODER_FILTER_1_DEPTH,
                         _block_size_3D_2);

  // Second conv2D layer (w/2, h/2)
  optimized1_conv2D(_out_max_pooling_1,
                    _encoder_filter_2,
                    _out_encoder_filter_2,
                    n,
                    width / 2,
                    height / 2,
                    ENCODER_FILTER_1_DEPTH,
                    ENCODER_FILTER_2_DEPTH,
                    _block_size_3D_2);
  optimized1_add_bias(_out_encoder_filter_2,
                      _encoder_bias_2,
                      _out_encoder_bias_2,
                      n,
                      width / 2,
                      height / 2,
                      ENCODER_FILTER_2_DEPTH,
                      _block_size_1D);
  optimized1_relu(_out_encoder_bias_2,
                  _out_encoder_relu_2,
                  n,
                  width / 2,
                  height / 2,
                  ENCODER_FILTER_2_DEPTH,
                  _block_size_1D);
  optimized1_max_pooling(_out_encoder_relu_2,
                         _out_max_pooling_2,
                         n,
                         width / 2,
                         height / 2,
                         ENCODER_FILTER_2_DEPTH,
                         _block_size_3D_3);

  // Decoder 1 (w/4, h/4)
  optimized1_conv2D(_out_max_pooling_2,
                    _decoder_filter_1,
                    _out_decoder_filter_1,
                    n,
                    width / 4,
                    height / 4,
                    ENCODER_FILTER_2_DEPTH,
                    DECODER_FILTER_1_DEPTH,
                    _block_size_3D_3);
  optimized1_add_bias(_out_decoder_filter_1,
                      _decoder_bias_1,
                      _out_decoder_bias_1,
                      n,
                      width / 4,
                      height / 4,
                      DECODER_FILTER_1_DEPTH,
                      _block_size_1D);
  optimized1_relu(_out_decoder_bias_1,
                  _out_decoder_relu_1,
                  n,
                  width / 4,
                  height / 4,
                  DECODER_FILTER_1_DEPTH,
                  _block_size_1D);
  optimized1_upsampling(_out_decoder_relu_1,
                        _out_upsampling_1,
                        n,
                        width / 4,
                        height / 4,
                        DECODER_FILTER_1_DEPTH,
                        _block_size_3D_2);

  // Decoder 2 (w/2, h/2)
  optimized1_conv2D(_out_upsampling_1,
                    _decoder_filter_2,
                    _out_decoder_filter_2,
                    n,
                    width / 2,
                    height / 2,
                    DECODER_FILTER_1_DEPTH,
                    DECODER_FILTER_2_DEPTH,
                    _block_size_3D_2);
  optimized1_add_bias(_out_decoder_filter_2,
                      _decoder_bias_2,
                      _out_decoder_bias_2,
                      n,
                      width / 2,
                      height / 2,
                      DECODER_FILTER_2_DEPTH,
                      _block_size_1D);
  optimized1_relu(_out_decoder_bias_2,
                  _out_decoder_relu_2,
                  n,
                  width / 2,
                  height / 2,
                  DECODER_FILTER_2_DEPTH,
                  _block_size_1D);
  optimized1_upsampling(_out_decoder_relu_2,
                        _out_upsampling_2,
                        n,
                        width / 2,
                        height / 2,
                        DECODER_FILTER_2_DEPTH,
                        _block_size_3D_1);

  // Decoder 3 (w, h)
  optimized1_conv2D(_out_upsampling_2,
                    _decoder_filter_3,
                    _out_decoder_filter_3,
                    n,
                    width,
                    height,
                    DECODER_FILTER_2_DEPTH,
                    DECODER_FILTER_3_DEPTH,
                    _block_size_3D_1);
  optimized1_add_bias(_out_decoder_filter_3,
                      _decoder_bias_3,
                      _out_decoder_bias_3,
                      n,
                      width,
                      height,
                      DECODER_FILTER_3_DEPTH,
                      _block_size_1D);

  CUDA_CHECK(cudaMemcpy(
      _res_data, _out_decoder_bias_3, size * sizeof(float), cudaMemcpyDeviceToDevice));
}

float Optimized1_Autoencoder::_fit_batch(
    float *data, int n, int width, int height, int depth, float learning_rate) {
  float *d_in = _d_in, *d_out = _d_out, *d_filter = _d_filter;

  _forward_pass(data, n, width, height, depth);

  float loss = optimized1_mse_loss(
      _batch_data, _res_data, n, width, height, depth, _block_size_1D);
  optimized1_mse_grad(
      _batch_data, _res_data, d_out, n, width, height, depth, _block_size_1D);

  // Decoder 3
  optimized1_bias_grad(
      d_out, d_in, n, width, height, DECODER_FILTER_3_DEPTH, _block_size_1D);
  optimized1_update_weight(
      _decoder_bias_3, d_in, DECODER_FILTER_3_DEPTH, learning_rate, _block_size_1D);

  optimized1_conv2D_grad(_out_upsampling_2,
                         d_out,
                         d_filter,
                         n,
                         width,
                         height,
                         DECODER_FILTER_2_DEPTH,
                         DECODER_FILTER_3_DEPTH,
                         _block_size_3D_1);
  optimized1_conv2D_backward(d_out,
                             _decoder_filter_3,
                             d_in,
                             n,
                             width,
                             height,
                             DECODER_FILTER_2_DEPTH,
                             DECODER_FILTER_3_DEPTH,
                             _block_size_3D_1);
  swap(d_out, d_in);
  optimized1_update_weight(_decoder_filter_3,
                           d_filter,
                           DECODER_FILTER_3_SIZE,
                           learning_rate,
                           _block_size_1D);

  // Decoder 2
  optimized1_upsampling_backward(
      d_out, d_in, n, width / 2, height / 2, DECODER_FILTER_2_DEPTH, _block_size_3D_2);
  optimized1_relu_backward(_out_decoder_bias_2,
                           d_in,
                           d_out,
                           n,
                           width / 2,
                           height / 2,
                           DECODER_FILTER_2_DEPTH,
                           _block_size_1D);

  optimized1_bias_grad(
      d_out, d_in, n, width / 2, height / 2, DECODER_FILTER_2_DEPTH, _block_size_1D);
  optimized1_update_weight(
      _decoder_bias_2, d_in, DECODER_FILTER_2_DEPTH, learning_rate, _block_size_1D);

  optimized1_conv2D_grad(_out_upsampling_1,
                         d_out,
                         d_filter,
                         n,
                         width / 2,
                         height / 2,
                         DECODER_FILTER_1_DEPTH,
                         DECODER_FILTER_2_DEPTH,
                         _block_size_3D_2);
  optimized1_conv2D_backward(d_out,
                             _decoder_filter_2,
                             d_in,
                             n,
                             width / 2,
                             height / 2,
                             DECODER_FILTER_1_DEPTH,
                             DECODER_FILTER_2_DEPTH,
                             _block_size_3D_2);
  swap(d_out, d_in);
  optimized1_update_weight(_decoder_filter_2,
                           d_filter,
                           DECODER_FILTER_2_SIZE,
                           learning_rate,
                           _block_size_1D);

  // Decoder 1
  optimized1_upsampling_backward(
      d_out, d_in, n, width / 4, height / 4, DECODER_FILTER_1_DEPTH, _block_size_3D_3);
  optimized1_relu_backward(_out_decoder_bias_1,
                           d_in,
                           d_out,
                           n,
                           width / 4,
                           height / 4,
                           DECODER_FILTER_1_DEPTH,
                           _block_size_1D);

  optimized1_bias_grad(
      d_out, d_in, n, width / 4, height / 4, DECODER_FILTER_1_DEPTH, _block_size_1D);
  optimized1_update_weight(
      _decoder_bias_1, d_in, DECODER_FILTER_1_DEPTH, learning_rate, _block_size_1D);

  optimized1_conv2D_grad(_out_max_pooling_2,
                         d_out,
                         d_filter,
                         n,
                         width / 4,
                         height / 4,
                         ENCODER_FILTER_2_DEPTH,
                         DECODER_FILTER_1_DEPTH,
                         _block_size_3D_3);
  optimized1_conv2D_backward(d_out,
                             _decoder_filter_1,
                             d_in,
                             n,
                             width / 4,
                             height / 4,
                             ENCODER_FILTER_2_DEPTH,
                             DECODER_FILTER_1_DEPTH,
                             _block_size_3D_3);
  swap(d_out, d_in);
  optimized1_update_weight(_decoder_filter_1,
                           d_filter,
                           DECODER_FILTER_1_SIZE,
                           learning_rate,
                           _block_size_1D);

  // Encoder 2
  optimized1_max_pooling_backward(_out_encoder_relu_2,
                                  d_out,
                                  d_in,
                                  n,
                                  width / 2,
                                  height / 2,
                                  ENCODER_FILTER_2_DEPTH,
                                  _block_size_3D_2);
  optimized1_relu_backward(_out_encoder_bias_2,
                           d_in,
                           d_out,
                           n,
                           width / 2,
                           height / 2,
                           ENCODER_FILTER_2_DEPTH,
                           _block_size_1D);

  optimized1_bias_grad(
      d_out, d_in, n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH, _block_size_1D);
  optimized1_update_weight(
      _encoder_bias_2, d_in, ENCODER_FILTER_2_DEPTH, learning_rate, _block_size_1D);

  optimized1_conv2D_grad(_out_max_pooling_1,
                         d_out,
                         d_filter,
                         n,
                         width / 2,
                         height / 2,
                         ENCODER_FILTER_1_DEPTH,
                         ENCODER_FILTER_2_DEPTH,
                         _block_size_3D_2);
  optimized1_conv2D_backward(d_out,
                             _encoder_filter_2,
                             d_in,
                             n,
                             width / 2,
                             height / 2,
                             ENCODER_FILTER_1_DEPTH,
                             ENCODER_FILTER_2_DEPTH,
                             _block_size_3D_2);
  swap(d_out, d_in);
  optimized1_update_weight(_encoder_filter_2,
                           d_filter,
                           ENCODER_FILTER_2_SIZE,
                           learning_rate,
                           _block_size_1D);

  // Encoder 1
  optimized1_max_pooling_backward(_out_encoder_relu_1,
                                  d_out,
                                  d_in,
                                  n,
                                  width,
                                  height,
                                  ENCODER_FILTER_1_DEPTH,
                                  _block_size_3D_1);
  optimized1_relu_backward(_out_encoder_bias_1,
                           d_in,
                           d_out,
                           n,
                           width,
                           height,
                           ENCODER_FILTER_1_DEPTH,
                           _block_size_3D_1);

  optimized1_bias_grad(
      d_out, d_in, n, width, height, ENCODER_FILTER_1_DEPTH, _block_size_1D);
  optimized1_update_weight(
      _encoder_bias_1, d_in, ENCODER_FILTER_1_DEPTH, learning_rate, _block_size_1D);

  optimized1_conv2D_grad(_batch_data,
                         d_out,
                         d_filter,
                         n,
                         width,
                         height,
                         depth,
                         ENCODER_FILTER_1_DEPTH,
                         _block_size_3D_1);
  optimized1_conv2D_backward(d_out,
                             _encoder_filter_1,
                             d_in,
                             n,
                             width,
                             height,
                             depth,
                             ENCODER_FILTER_1_DEPTH,
                             _block_size_3D_1);
  swap(d_out, d_in);
  optimized1_update_weight(_encoder_filter_1,
                           d_filter,
                           ENCODER_FILTER_1_SIZE,
                           learning_rate,
                           _block_size_1D);

  return loss;
}

void Optimized1_Autoencoder::fit(const Optimized_Dataset &dataset,
                                 int                      n_epoch,
                                 int                      batch_size,
                                 float                    learning_rate,
                                 int                      checkpoint,
                                 const char              *output_dir) {
  int n = dataset.n, width = dataset.width, height = dataset.height,
      depth      = dataset.depth;
  int n_batch    = (dataset.n - 1) / batch_size + 1;
  int image_size = width * height * depth;

  _allocate_output_mem(batch_size, dataset.width, dataset.height);
  filesystem::create_directories(output_dir);

  Timer timer;
  float total_time = 0;

  printf(
      "Training Optimized Autoencoder (1st version) for %d epochs with batch size %d "
      "and learning rate %.4f\n",
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

    if (checkpoint > 0 && epoch % checkpoint == 0) {
      stringstream builder;
      builder << output_dir << '/' << "optimized1_autoencoder_" << epoch << ".bin";
      save_parameters(builder.str().c_str());
    }
  }
  puts("========================TRAINING END========================");

  _deallocate_output_mem();

  stringstream builder;
  builder << output_dir << '/' << "optimized1_autoencoder.bin";
  save_parameters(builder.str().c_str());

  printf("\nTotal time: %s (ms), Loss: %.4f\n",
         format_time(total_time).c_str(),
         eval(dataset));
}

Optimized_Dataset
Optimized1_Autoencoder::encode(const Optimized_Dataset &dataset) const {
  int width = dataset.width, height = dataset.height, depth = dataset.depth,
      n                  = dataset.n;
  int n_batch            = (n - 1) / ENCODE_BATCH_SIZE + 1;
  int image_size         = width * height * depth;
  int encoded_image_size = width / 4 * height / 4 * ENCODER_FILTER_2_DEPTH;
  int out_offset         = 0;

  Optimized_Dataset res(n, width / 4, height / 4, ENCODER_FILTER_2_DEPTH);

  float *a, *b;
  CUDA_CHECK(cudaMalloc(
      &a, ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &b, ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH * sizeof(float)));

  for (int i = 0; i < n_batch; ++i) {
    int in_offset      = i * ENCODE_BATCH_SIZE;
    int cur_batch_size = min(ENCODE_BATCH_SIZE, n - in_offset);
    CUDA_CHECK(cudaMemcpy(b,
                          dataset.data + in_offset * image_size,
                          cur_batch_size * image_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    optimized1_conv2D(b,
                      _encoder_filter_1,
                      a,
                      cur_batch_size,
                      width,
                      height,
                      depth,
                      ENCODER_FILTER_1_DEPTH,
                      _block_size_3D_1);
    optimized1_add_bias(a,
                        _encoder_bias_1,
                        b,
                        cur_batch_size,
                        width,
                        height,
                        ENCODER_FILTER_1_DEPTH,
                        _block_size_1D);
    optimized1_relu(
        b, a, cur_batch_size, width, height, ENCODER_FILTER_1_DEPTH, _block_size_1D);
    optimized1_max_pooling(
        a, b, cur_batch_size, width, height, ENCODER_FILTER_1_DEPTH, _block_size_3D_2);

    optimized1_conv2D(b,
                      _encoder_filter_2,
                      a,
                      cur_batch_size,
                      width / 2,
                      height / 2,
                      ENCODER_FILTER_1_DEPTH,
                      ENCODER_FILTER_2_DEPTH,
                      _block_size_3D_2);
    optimized1_add_bias(a,
                        _encoder_bias_2,
                        b,
                        cur_batch_size,
                        width / 2,
                        height / 2,
                        ENCODER_FILTER_2_DEPTH,
                        _block_size_1D);
    optimized1_relu(b,
                    a,
                    cur_batch_size,
                    width / 2,
                    height / 2,
                    ENCODER_FILTER_2_DEPTH,
                    _block_size_1D);
    optimized1_max_pooling(a,
                           b,
                           cur_batch_size,
                           width / 2,
                           height / 2,
                           ENCODER_FILTER_2_DEPTH,
                           _block_size_3D_3);

    CUDA_CHECK(cudaMemcpy(res.data + out_offset * encoded_image_size,
                          b,
                          cur_batch_size * encoded_image_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    out_offset += cur_batch_size;
  }
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  memcpy(res.labels, dataset.labels, n * sizeof(int));
  return res;
}

Optimized_Dataset
Optimized1_Autoencoder::decode(const Optimized_Dataset &dataset) const {
  int width = dataset.width, height = dataset.height, depth = dataset.depth,
      n                  = dataset.n;
  int n_batch            = (n - 1) / ENCODE_BATCH_SIZE + 1;
  int image_size         = width * height * depth;
  int decoded_image_size = 4 * width * 4 * height * DECODER_FILTER_3_DEPTH;
  int out_offset         = 0;

  Optimized_Dataset res(dataset.n, width * 4, height * 4, DECODER_FILTER_3_DEPTH);
  float            *a, *b;
  CUDA_CHECK(cudaMalloc(
      &a,
      ENCODE_BATCH_SIZE * 4 * width * 4 * height * MAX_FILTER_DEPTH * sizeof(float)));
  CUDA_CHECK(cudaMalloc(
      &b,
      ENCODE_BATCH_SIZE * 4 * width * 4 * height * MAX_FILTER_DEPTH * sizeof(float)));

  for (int i = 0; i < n_batch; ++i) {
    int in_offset      = i * ENCODE_BATCH_SIZE;
    int cur_batch_size = min(ENCODE_BATCH_SIZE, n - in_offset);
    CUDA_CHECK(cudaMemcpy(b,
                          dataset.data + in_offset * image_size,
                          cur_batch_size * image_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    optimized1_conv2D(b,
                      _decoder_filter_1,
                      a,
                      cur_batch_size,
                      width,
                      height,
                      depth,
                      DECODER_FILTER_1_DEPTH,
                      _block_size_3D_3);
    optimized1_add_bias(a,
                        _decoder_bias_1,
                        b,
                        cur_batch_size,
                        width,
                        height,
                        DECODER_FILTER_1_DEPTH,
                        _block_size_1D);
    optimized1_relu(
        b, a, cur_batch_size, width, height, DECODER_FILTER_1_DEPTH, _block_size_1D);
    optimized1_upsampling(
        a, b, cur_batch_size, width, height, DECODER_FILTER_1_DEPTH, _block_size_3D_3);

    optimized1_conv2D(b,
                      _decoder_filter_2,
                      a,
                      cur_batch_size,
                      width * 2,
                      height * 2,
                      DECODER_FILTER_1_DEPTH,
                      DECODER_FILTER_2_DEPTH,
                      _block_size_3D_2);
    optimized1_add_bias(a,
                        _decoder_bias_2,
                        b,
                        cur_batch_size,
                        width * 2,
                        height * 2,
                        DECODER_FILTER_2_DEPTH,
                        _block_size_1D);
    optimized1_relu(b,
                    a,
                    cur_batch_size,
                    width * 2,
                    height * 2,
                    DECODER_FILTER_2_DEPTH,
                    _block_size_1D);
    optimized1_upsampling(a,
                          b,
                          cur_batch_size,
                          width * 2,
                          height * 2,
                          DECODER_FILTER_2_DEPTH,
                          _block_size_3D_1);

    optimized1_conv2D(b,
                      _decoder_filter_3,
                      a,
                      cur_batch_size,
                      width * 4,
                      height * 4,
                      DECODER_FILTER_2_DEPTH,
                      DECODER_FILTER_3_DEPTH,
                      _block_size_3D_1);
    optimized1_add_bias(a,
                        _decoder_bias_3,
                        b,
                        cur_batch_size,
                        width * 4,
                        height * 4,
                        DECODER_FILTER_3_DEPTH,
                        _block_size_1D);

    CUDA_CHECK(cudaMemcpy(res.data + out_offset * decoded_image_size,
                          b,
                          cur_batch_size * decoded_image_size * sizeof(float),
                          cudaMemcpyDeviceToHost));
    out_offset += cur_batch_size;
  }
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  memcpy(res.labels, dataset.labels, dataset.n * sizeof(int));
  return res;
}

float Optimized1_Autoencoder::eval(const Optimized_Dataset &dataset) const {
  int n = dataset.n, width = dataset.width, height = dataset.height,
      depth              = dataset.depth;
  int               size = n * width * height * depth;
  Optimized_Dataset res  = decode(encode(dataset));

  float *a, *b;
  CUDA_CHECK(cudaMalloc(&a, size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&b, size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(a, dataset.data, size * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b, res.data, size * sizeof(float), cudaMemcpyHostToDevice));
  float val = optimized1_mse_loss(a, b, n, width, height, depth, _block_size_1D);

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  return val;
}

void Optimized1_Autoencoder::save_parameters(const char *filename) const {
  ofstream buffer(filename, ios::out | ios::binary);
  if (!buffer.is_open()) {
    printf("Unable to open the file %s.\n", filename);
    return;
  }

  // Uses STATIC write_data (no conflict)
  write_data(buffer, _encoder_filter_1, ENCODER_FILTER_1_SIZE * sizeof(float));
  write_data(buffer, _encoder_bias_1, ENCODER_FILTER_1_DEPTH * sizeof(float));
  write_data(buffer, _encoder_filter_2, ENCODER_FILTER_2_SIZE * sizeof(float));
  write_data(buffer, _encoder_bias_2, ENCODER_FILTER_2_DEPTH * sizeof(float));
  write_data(buffer, _decoder_filter_1, DECODER_FILTER_1_SIZE * sizeof(float));
  write_data(buffer, _decoder_bias_1, DECODER_FILTER_1_DEPTH * sizeof(float));
  write_data(buffer, _decoder_filter_2, DECODER_FILTER_2_SIZE * sizeof(float));
  write_data(buffer, _decoder_bias_2, DECODER_FILTER_2_DEPTH * sizeof(float));
  write_data(buffer, _decoder_filter_3, DECODER_FILTER_3_SIZE * sizeof(float));
  write_data(buffer, _decoder_bias_3, DECODER_FILTER_3_DEPTH * sizeof(float));

  buffer.close();
  printf("✓ Optimized1 Autoencoder parameters saved successfully to %s\n", filename);
}

void Optimized1_Autoencoder::load_parameters(const char *filename) {
  ifstream buffer(filename, ios::in | ios::binary);
  if (!buffer.is_open()) {
    fprintf(stderr, "Error: Cannot open file %s for reading\n", filename);
    return;
  }

  // Uses STATIC read_data (no conflict)
  read_data(buffer, _encoder_filter_1, ENCODER_FILTER_1_SIZE * sizeof(float));
  read_data(buffer, _encoder_bias_1, ENCODER_FILTER_1_DEPTH * sizeof(float));
  read_data(buffer, _encoder_filter_2, ENCODER_FILTER_2_SIZE * sizeof(float));
  read_data(buffer, _encoder_bias_2, ENCODER_FILTER_2_DEPTH * sizeof(float));
  read_data(buffer, _decoder_filter_1, DECODER_FILTER_1_SIZE * sizeof(float));
  read_data(buffer, _decoder_bias_1, DECODER_FILTER_1_DEPTH * sizeof(float));
  read_data(buffer, _decoder_filter_2, DECODER_FILTER_2_SIZE * sizeof(float));
  read_data(buffer, _decoder_bias_2, DECODER_FILTER_2_DEPTH * sizeof(float));
  read_data(buffer, _decoder_filter_3, DECODER_FILTER_3_SIZE * sizeof(float));
  read_data(buffer, _decoder_bias_3, DECODER_FILTER_3_DEPTH * sizeof(float));

  buffer.close();
  printf("✓ Optimized1 Autoencoder parameters loaded successfully from %s\n", filename);
}