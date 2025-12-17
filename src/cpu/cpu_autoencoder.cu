#include "cpu_autoencoder.h"

string AUTOENCODER_FILENAME = "autoencoder.bin";

/**
 * @brief Generate a random array with elements between -0.01 and 0.01
 *
 * @param arr The array
 * @param n The number of elements
 */
void generate_array(const unique_ptr<float[]> &arr, int n) {
  float *ptr = arr.get();
  for (int i = 0; i < n; ++i)
    ptr[i] = 0.2f * (rand() - RAND_MAX / 2) / RAND_MAX;
}

/**
 * @brief Read data from a buffer
 *
 * @param buffer The buffer
 * @param data The data
 * @param size Number of bytes to read
 */
void read_data(ifstream &buffer, const unique_ptr<float[]> &data, int size) {
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
void write_data(ostream &buffer, const unique_ptr<float[]> &data, int size) {
  const char *ptr = reinterpret_cast<const char *>(data.get());
  buffer.write(ptr, size);
}

Cpu_Autoencoder::Cpu_Autoencoder() {
  _allocate_mem();

  // Random init
  srand(time(0));

  generate_array(_encoder_filter_1, ENCODER_FILTER_1_SIZE);
  generate_array(_encoder_bias_1, ENCODER_FILTER_1_DEPTH);

  generate_array(_encoder_filter_2, ENCODER_FILTER_2_SIZE);
  generate_array(_encoder_bias_2, ENCODER_FILTER_2_DEPTH);

  generate_array(_decoder_filter_1, DECODER_FILTER_1_SIZE);
  generate_array(_decoder_bias_1, DECODER_FILTER_1_DEPTH);

  generate_array(_decoder_filter_2, DECODER_FILTER_2_SIZE);
  generate_array(_decoder_bias_2, DECODER_FILTER_2_DEPTH);

  generate_array(_decoder_filter_3, DECODER_FILTER_3_SIZE);
  generate_array(_decoder_bias_3, DECODER_FILTER_3_DEPTH);
};

Cpu_Autoencoder::Cpu_Autoencoder(const char *filename) {
  _allocate_mem();

  // Read from file
  ifstream buffer(filename, ios::in | ios::binary);

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

  // Read third decoder conv2D layer
  read_data(buffer, _decoder_filter_3, DECODER_FILTER_3_SIZE * sizeof(float));
  read_data(buffer, _decoder_bias_3, DECODER_FILTER_3_DEPTH * sizeof(float));
};

void Cpu_Autoencoder::_allocate_mem() {
  _encoder_filter_1 = make_unique<float[]>(ENCODER_FILTER_1_SIZE);
  _encoder_bias_1   = make_unique<float[]>(ENCODER_FILTER_1_DEPTH);

  _encoder_filter_2 = make_unique<float[]>(ENCODER_FILTER_2_SIZE);
  _encoder_bias_2   = make_unique<float[]>(ENCODER_FILTER_2_DEPTH);

  _decoder_filter_1 = make_unique<float[]>(DECODER_FILTER_1_SIZE);
  _decoder_bias_1   = make_unique<float[]>(DECODER_FILTER_1_DEPTH);

  _decoder_filter_2 = make_unique<float[]>(DECODER_FILTER_2_SIZE);
  _decoder_bias_2   = make_unique<float[]>(DECODER_FILTER_2_DEPTH);

  _decoder_filter_3 = make_unique<float[]>(DECODER_FILTER_3_SIZE);
  _decoder_bias_3   = make_unique<float[]>(DECODER_FILTER_3_DEPTH);
}

Dataset Cpu_Autoencoder::_encode_save_output(const Dataset &dataset) {
  int n = dataset.n, width = dataset.width, height = dataset.height,
      depth = dataset.depth;

  // First conv2D layer
  cpu_conv2D(dataset.get_data(),
             _encoder_filter_1.get(),
             _out_encoder_filter_1.get(),
             n,
             width,
             height,
             depth,
             ENCODER_FILTER_1_DEPTH);

  // Dim: n * w * w * 256
  cpu_add_bias(_out_encoder_filter_1.get(),
               _encoder_bias_1.get(),
               _out_encoder_bias_1.get(),
               n,
               width,
               height,
               ENCODER_FILTER_1_DEPTH);

  // ReLU layer
  cpu_relu(_out_encoder_bias_1.get(),
           _out_encoder_relu_1.get(),
           n,
           width,
           height,
           ENCODER_FILTER_1_DEPTH);

  // First max pooling layer
  cpu_avg_pooling(_out_encoder_relu_1.get(),
                  _out_avg_pooling_1.get(),
                  n,
                  width,
                  height,
                  ENCODER_FILTER_1_DEPTH);

  // Dim: n * w/2 * w/2 * 256
  // Second conv2D layer
  cpu_conv2D(_out_avg_pooling_1.get(),
             _encoder_filter_2.get(),
             _out_encoder_filter_2.get(),
             n,
             width / 2,
             height / 2,
             ENCODER_FILTER_1_DEPTH,
             ENCODER_FILTER_2_DEPTH);

  // Dim: n * w/2 * w/2 * 128
  cpu_add_bias(_out_encoder_filter_2.get(),
               _encoder_bias_2.get(),
               _out_encoder_bias_2.get(),
               n,
               width / 2,
               height / 2,
               ENCODER_FILTER_2_DEPTH);

  // ReLU layer
  cpu_relu(_out_encoder_bias_2.get(),
           _out_encoder_relu_2.get(),
           n,
           width / 2,
           height / 2,
           ENCODER_FILTER_2_DEPTH);

  // Second max pooling layer
  cpu_avg_pooling(_out_encoder_relu_2.get(),
                  _out_avg_pooling_2.get(),
                  n,
                  width / 2,
                  height / 2,
                  ENCODER_FILTER_2_DEPTH);

  // Return the result (Dim: n * w/4 * w/4 * 128)
  Dataset res(n, width / 4, height / 4, ENCODER_FILTER_2_DEPTH);
  memcpy(res.get_data(),
         _out_avg_pooling_2.get(),
         n * (width / 4) * (height / 4) * ENCODER_FILTER_2_DEPTH * sizeof(float));

  return res;
}

Dataset Cpu_Autoencoder::_decode_save_output(const Dataset &dataset) {
  int n = dataset.n, width = dataset.width, height = dataset.height,
      depth = dataset.depth;

  // First conv2D layer
  cpu_conv2D(dataset.get_data(),
             _decoder_filter_1.get(),
             _out_decoder_filter_1.get(),
             n,
             width,
             height,
             depth,
             DECODER_FILTER_1_DEPTH);

  // Dim: n * w * w * 128
  cpu_add_bias(_out_decoder_filter_1.get(),
               _decoder_bias_1.get(),
               _out_decoder_bias_1.get(),
               n,
               width,
               height,
               DECODER_FILTER_1_DEPTH);

  // ReLU layer
  cpu_relu(_out_decoder_bias_1.get(),
           _out_decoder_relu_1.get(),
           n,
           width,
           height,
           DECODER_FILTER_1_DEPTH);

  // First upsampling layer
  cpu_upsampling(_out_decoder_relu_1.get(),
                 _out_upsampling_1.get(),
                 n,
                 width,
                 height,
                 DECODER_FILTER_1_DEPTH);

  // Dim: n * 2w * 2w * 256
  // Second conv2D layer
  cpu_conv2D(_out_upsampling_1.get(),
             _decoder_filter_2.get(),
             _out_decoder_filter_2.get(),
             n,
             2 * width,
             2 * height,
             DECODER_FILTER_1_DEPTH,
             DECODER_FILTER_2_DEPTH);

  // Dim: n * 2w * 2w * 256
  cpu_add_bias(_out_decoder_filter_2.get(),
               _decoder_bias_2.get(),
               _out_decoder_bias_2.get(),
               n,
               2 * width,
               2 * height,
               DECODER_FILTER_2_DEPTH);

  // ReLU layer
  cpu_relu(_out_decoder_bias_2.get(),
           _out_decoder_relu_2.get(),
           n,
           2 * width,
           2 * height,
           DECODER_FILTER_2_DEPTH);

  // Second upsampling layer
  cpu_upsampling(_out_decoder_relu_2.get(),
                 _out_upsampling_2.get(),
                 n,
                 2 * width,
                 2 * height,
                 DECODER_FILTER_2_DEPTH);

  // Dim: n * 4w * 4w * 256
  // Third conv2D layer
  cpu_conv2D(_out_upsampling_2.get(),
             _decoder_filter_3.get(),
             _out_decoder_filter_3.get(),
             n,
             4 * width,
             4 * height,
             DECODER_FILTER_2_DEPTH,
             DECODER_FILTER_3_DEPTH);

  // Dim: n * 4w * 4w * 3
  cpu_add_bias(_out_decoder_filter_3.get(),
               _decoder_bias_3.get(),
               _out_decoder_bias_3.get(),
               n,
               4 * width,
               4 * height,
               DECODER_FILTER_3_DEPTH);

  // Return the result (Dim: n * 4w * 4h * 3)
  Dataset res(n, 4 * width, 4 * height, DECODER_FILTER_3_DEPTH);
  memcpy(res.get_data(),
         _out_decoder_bias_3.get(),
         n * 4 * width * 4 * height * DECODER_FILTER_3_DEPTH * sizeof(float));
  return res;
}

void Cpu_Autoencoder::_allocate_output_mem(int n, int width, int height) {
  int n_pixel = n * width * height;

  _out_encoder_filter_1 = make_unique<float[]>(n_pixel * ENCODER_FILTER_1_DEPTH);
  _out_encoder_bias_1   = make_unique<float[]>(n_pixel * ENCODER_FILTER_1_DEPTH);
  _out_encoder_relu_1   = make_unique<float[]>(n_pixel * ENCODER_FILTER_1_DEPTH);
  _out_avg_pooling_1    = make_unique<float[]>(n_pixel * ENCODER_FILTER_1_DEPTH / 4);

  _out_encoder_filter_2 = make_unique<float[]>(n_pixel * ENCODER_FILTER_2_DEPTH);
  _out_encoder_bias_2   = make_unique<float[]>(n_pixel * ENCODER_FILTER_2_DEPTH);
  _out_encoder_relu_2   = make_unique<float[]>(n_pixel * ENCODER_FILTER_2_DEPTH);
  _out_avg_pooling_2    = make_unique<float[]>(n_pixel * ENCODER_FILTER_2_DEPTH / 16);

  _out_decoder_filter_1 = make_unique<float[]>(n_pixel * DECODER_FILTER_1_DEPTH / 16);
  _out_decoder_bias_1   = make_unique<float[]>(n_pixel * DECODER_FILTER_1_DEPTH / 16);
  _out_decoder_relu_1   = make_unique<float[]>(n_pixel * DECODER_FILTER_1_DEPTH / 16);
  _out_upsampling_1     = make_unique<float[]>(n_pixel * DECODER_FILTER_1_DEPTH / 4);

  _out_decoder_filter_2 = make_unique<float[]>(n_pixel * DECODER_FILTER_2_DEPTH / 4);
  _out_decoder_bias_2   = make_unique<float[]>(n_pixel * DECODER_FILTER_2_DEPTH / 4);
  _out_decoder_relu_2   = make_unique<float[]>(n_pixel * DECODER_FILTER_2_DEPTH / 4);
  _out_upsampling_2     = make_unique<float[]>(n_pixel * DECODER_FILTER_2_DEPTH);

  _out_decoder_filter_3 = make_unique<float[]>(n_pixel * DECODER_FILTER_3_DEPTH);
  _out_decoder_bias_3   = make_unique<float[]>(n_pixel * DECODER_FILTER_3_DEPTH);

  static constexpr int FILTER_SIZES[]  = { ENCODER_FILTER_1_SIZE,
                                           ENCODER_FILTER_2_SIZE,
                                           DECODER_FILTER_1_SIZE,
                                           DECODER_FILTER_2_SIZE,
                                           DECODER_FILTER_3_SIZE };
  constexpr int        MAX_FILTER_SIZE = *max_element(FILTER_SIZES, FILTER_SIZES + 5);

  _d_in = make_unique<float[]>(n * width * height * MAX_FILTER_DEPTH);

  _d_out    = make_unique<float[]>(n * width * height * MAX_FILTER_DEPTH);
  _d_filter = make_unique<float[]>(MAX_FILTER_SIZE);
}

void Cpu_Autoencoder::_deallocate_output_mem() {
  _out_encoder_filter_1 = 0;
  _out_encoder_bias_1   = 0;
  _out_encoder_relu_1   = 0;
  _out_avg_pooling_1    = 0;

  _out_encoder_filter_2 = 0;
  _out_encoder_bias_2   = 0;
  _out_encoder_relu_2   = 0;
  _out_avg_pooling_2    = 0;

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
  int     n = batch.n, width = batch.width, height = batch.height, depth = batch.depth;
  float  *d_in = _d_in.get(), *d_out = _d_out.get(), *d_filter = _d_filter.get();
  Dataset res = _decode_save_output(_encode_save_output(batch));

  // Calculate loss before backprop
  float loss = cpu_mse_loss(batch.get_data(), res.get_data(), n, width, height, depth);

  // Get loss gradient
  cpu_mse_grad(batch.get_data(), res.get_data(), d_out, n, width, height, depth);

  // Update weight for the last conv2D layer
  // Update bias
  cpu_bias_grad(d_out, d_in, n, width, height, DECODER_FILTER_3_DEPTH);

  cpu_update_weight(_decoder_bias_3.get(), d_in, DECODER_FILTER_3_DEPTH, learning_rate);

  // Update filter
  cpu_conv2D_grad(_out_upsampling_2.get(),
                  d_out,
                  d_filter,
                  n,
                  width,
                  height,
                  DECODER_FILTER_2_DEPTH,
                  DECODER_FILTER_3_DEPTH);

  // Pass delta backwards
  cpu_conv2D(d_out,
             _decoder_filter_3.get(),
             d_in,
             n,
             width,
             height,
             DECODER_FILTER_2_DEPTH,
             DECODER_FILTER_3_DEPTH);

  // Swap d_out and d_in
  swap(d_out, d_in);

  // Update weight
  cpu_update_weight(
      _decoder_filter_3.get(), d_filter, DECODER_FILTER_3_SIZE, learning_rate);

  // Pass through upsampling (dim: n * w/2 * w/2 * 256)
  cpu_upsampling_backward(
      d_out, d_in, n, width / 2, height / 2, DECODER_FILTER_2_DEPTH);

  // Pass through ReLU (d_in and d_out swapped)
  cpu_relu_backward(_out_decoder_bias_2.get(),
                    d_in,
                    d_out,
                    n,
                    width / 2,
                    height / 2,
                    DECODER_FILTER_2_DEPTH);

  // Second conv2D layer
  cpu_bias_grad(d_out, d_in, n, width / 2, height / 2, DECODER_FILTER_2_DEPTH);

  cpu_update_weight(_decoder_bias_2.get(), d_in, DECODER_FILTER_2_DEPTH, learning_rate);

  cpu_conv2D_grad(_out_upsampling_1.get(),
                  d_out,
                  d_filter,
                  n,
                  width / 2,
                  height / 2,
                  DECODER_FILTER_1_DEPTH,
                  DECODER_FILTER_2_DEPTH);

  cpu_conv2D(d_out,
             _decoder_filter_2.get(),
             d_in,
             n,
             width / 2,
             height / 2,
             DECODER_FILTER_1_DEPTH,
             DECODER_FILTER_2_DEPTH);

  swap(d_out, d_in);
  cpu_update_weight(
      _decoder_filter_2.get(), d_filter, DECODER_FILTER_2_SIZE, learning_rate);

  // Upsampling (dim: n * w/4 * w/4 * 128)
  cpu_upsampling_backward(
      d_out, d_in, n, width / 4, height / 4, DECODER_FILTER_1_DEPTH);

  // ReLU
  cpu_relu_backward(_out_decoder_bias_1.get(),
                    d_in,
                    d_out,
                    n,
                    width / 4,
                    height / 4,
                    DECODER_FILTER_1_DEPTH);

  // Third Conv2D
  cpu_bias_grad(d_out, d_in, n, width / 4, height / 4, DECODER_FILTER_1_DEPTH);

  cpu_update_weight(_decoder_bias_1.get(), d_in, DECODER_FILTER_1_DEPTH, learning_rate);

  cpu_conv2D_grad(_out_avg_pooling_2.get(),
                  d_out,
                  d_filter,
                  n,
                  width / 4,
                  height / 4,
                  ENCODER_FILTER_2_DEPTH,
                  DECODER_FILTER_1_DEPTH);

  cpu_conv2D(d_out,
             _decoder_filter_1.get(),
             d_in,
             n,
             width / 4,
             height / 4,
             ENCODER_FILTER_2_DEPTH,
             DECODER_FILTER_1_DEPTH);

  swap(d_out, d_in);
  cpu_update_weight(
      _decoder_filter_1.get(), d_filter, DECODER_FILTER_1_SIZE, learning_rate);

  // Max pooling backwards (dim: n * w/2 * w/2 * 128)
  cpu_avg_pooling_backward(
      d_out, d_in, n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH);

  cpu_relu_backward(_out_encoder_bias_2.get(),
                    d_in,
                    d_out,
                    n,
                    width / 2,
                    height / 2,
                    ENCODER_FILTER_2_DEPTH);

  // Forth conv2D
  cpu_bias_grad(d_out, d_in, n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH);

  cpu_update_weight(_encoder_bias_2.get(), d_in, ENCODER_FILTER_2_DEPTH, learning_rate);

  cpu_conv2D_grad(_out_avg_pooling_1.get(),
                  d_out,
                  d_filter,
                  n,
                  width / 2,
                  height / 2,
                  ENCODER_FILTER_1_DEPTH,
                  ENCODER_FILTER_2_DEPTH);

  cpu_conv2D(d_out,
             _encoder_filter_2.get(),
             d_in,
             n,
             width / 2,
             height / 2,
             ENCODER_FILTER_1_DEPTH,
             ENCODER_FILTER_2_DEPTH);

  swap(d_out, d_in);
  cpu_update_weight(
      _encoder_filter_2.get(), d_filter, ENCODER_FILTER_2_SIZE, learning_rate);

  cpu_avg_pooling_backward(d_out, d_in, n, width, height, ENCODER_FILTER_1_DEPTH);

  cpu_relu_backward(
      _out_encoder_bias_1.get(), d_in, d_out, n, width, height, ENCODER_FILTER_1_DEPTH);

  // Fifth conv2D
  cpu_bias_grad(d_out, d_in, n, width, height, ENCODER_FILTER_1_DEPTH);

  cpu_update_weight(_encoder_bias_1.get(), d_in, ENCODER_FILTER_1_DEPTH, learning_rate);

  cpu_conv2D_grad(batch.get_data(),
                  d_out,
                  d_filter,
                  n,
                  width,
                  height,
                  depth,
                  ENCODER_FILTER_1_DEPTH);

  cpu_conv2D(d_out,
             _encoder_filter_1.get(),
             d_in,
             n,
             width,
             height,
             depth,
             ENCODER_FILTER_1_DEPTH);

  swap(d_out, d_in);
  cpu_update_weight(
      _encoder_filter_1.get(), d_filter, ENCODER_FILTER_1_SIZE, learning_rate);

  return loss;
}

void Cpu_Autoencoder::fit(const Dataset &dataset,
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

  Timer timer;
  float total_time = 0;

  puts("=======================TRAINING START=======================");
  for (int epoch = 1; epoch <= n_epoch; ++epoch) {
    printf("Epoch %d:\n", epoch);
    Progress_Bar bar(batches.size(), "Batch");
    bar.update();

    int   total      = 0;
    float total_loss = 0;
    float epoch_time = 0;
    for (const Dataset &batch : batches) {
      total += batch.n;
      timer.start();
      float batch_loss = _fit_batch(batch, learning_rate);
      timer.stop();
      total_loss += batch_loss * batch.n;
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
      builder << output_dir << '/' << "cpu_autoencoder_" << epoch << ".bin";
      save_parameters(builder.str().c_str());
    }
  }

  puts("========================TRAINING END========================");

  // Deallocate memory to remove unused memory
  _deallocate_output_mem();

  // Save models param
  stringstream builder;
  builder << output_dir << '/' << "cpu_autoencoder_.bin";
  save_parameters(builder.str().c_str());

  printf("\nTotal time: %s (ms), Loss: %.4f\n",
         format_time(total_time).c_str(),
         eval(dataset));
}

Dataset Cpu_Autoencoder::encode(const Dataset &dataset) const {
  // Encode by batches to use less memory
  int width = dataset.width, height = dataset.height, depth = dataset.depth;
  int encoded_image_bytes =
      (width / 4) * (height / 4) * ENCODER_FILTER_2_DEPTH * sizeof(float);
  int offset = 0;

  vector<Dataset> batches = create_minibatches(dataset, ENCODE_BATCH_SIZE);
  Dataset         res(dataset.n, width / 4, height / 4, ENCODER_FILTER_2_DEPTH);

  // Placeholder, alternating
  unique_ptr<float[]> a =
      make_unique<float[]>(ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH);
  unique_ptr<float[]> b =
      make_unique<float[]>(ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH);

  for (int i = 0; i < batches.size(); ++i) {
    int n = batches[i].n;

    // First conv2D
    cpu_conv2D(batches[i].get_data(),
               _encoder_filter_1.get(),
               a.get(),
               n,
               width,
               height,
               depth,
               ENCODER_FILTER_1_DEPTH);

    // Add bias
    cpu_add_bias(a.get(),
                 _encoder_bias_1.get(),
                 b.get(),
                 n,
                 width,
                 height,
                 ENCODER_FILTER_1_DEPTH);

    // ReLU
    cpu_relu(b.get(), a.get(), n, width, height, ENCODER_FILTER_1_DEPTH);

    // Max pooling
    cpu_avg_pooling(a.get(), b.get(), n, width, height, ENCODER_FILTER_1_DEPTH);

    // Second conv2D
    cpu_conv2D(b.get(),
               _encoder_filter_2.get(),
               a.get(),
               n,
               width / 2,
               height / 2,
               ENCODER_FILTER_1_DEPTH,
               ENCODER_FILTER_2_DEPTH);

    cpu_add_bias(a.get(),
                 _encoder_bias_2.get(),
                 b.get(),
                 n,
                 width / 2,
                 height / 2,
                 ENCODER_FILTER_2_DEPTH);

    // Second ReLU
    cpu_relu(b.get(), a.get(), n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH);

    // Second max pooling
    cpu_avg_pooling(a.get(), b.get(), n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH);

    // Copy batch
    memcpy(res.get_data() + offset, b.get(), n * encoded_image_bytes);
    offset += n * encoded_image_bytes;
  }

  // Copy labels
  memcpy(res.get_labels(), dataset.get_labels(), dataset.n * sizeof(int));
  return res;
}

Dataset Cpu_Autoencoder::decode(const Dataset &dataset) const {
  int width = dataset.width, height = dataset.height, depth = dataset.depth;
  int encoded_image_bytes =
      4 * width * 4 * height * DECODER_FILTER_3_DEPTH * sizeof(float);
  int offset = 0;

  vector<Dataset> batches = create_minibatches(dataset, ENCODE_BATCH_SIZE);
  Dataset         res(dataset.n, width * 4, height * 4, DECODER_FILTER_3_DEPTH);

  // Placeholder, alternating
  unique_ptr<float[]> a =
      make_unique<float[]>(ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH);
  unique_ptr<float[]> b =
      make_unique<float[]>(ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH);

  for (int i = 0; i < batches.size(); ++i) {
    int n = batches[i].n;

    // First conv2D
    cpu_conv2D(batches[i].get_data(),
               _decoder_filter_1.get(),
               a.get(),
               n,
               width,
               height,
               depth,
               DECODER_FILTER_1_DEPTH);

    // Add bias
    cpu_add_bias(a.get(),
                 _decoder_bias_1.get(),
                 b.get(),
                 n,
                 width,
                 height,
                 DECODER_FILTER_1_DEPTH);

    // ReLU
    cpu_relu(b.get(), a.get(), n, width, height, DECODER_FILTER_1_DEPTH);

    // Upsampling
    cpu_upsampling(a.get(), b.get(), n, width, height, DECODER_FILTER_1_DEPTH);

    // Second conv2D
    cpu_conv2D(b.get(),
               _decoder_filter_2.get(),
               a.get(),
               n,
               width * 2,
               height * 2,
               DECODER_FILTER_1_DEPTH,
               DECODER_FILTER_2_DEPTH);

    cpu_add_bias(a.get(),
                 _decoder_bias_2.get(),
                 b.get(),
                 n,
                 width * 2,
                 height * 2,
                 DECODER_FILTER_2_DEPTH);

    // Second ReLU
    cpu_relu(b.get(), a.get(), n, width * 2, height * 2, DECODER_FILTER_2_DEPTH);

    // Second upsampling
    cpu_upsampling(a.get(), b.get(), n, width * 2, height * 2, DECODER_FILTER_2_DEPTH);

    // Third conv2D
    cpu_conv2D(b.get(),
               _decoder_filter_3.get(),
               a.get(),
               n,
               width * 4,
               height * 4,
               DECODER_FILTER_2_DEPTH,
               DECODER_FILTER_3_DEPTH);

    cpu_add_bias(a.get(),
                 _decoder_bias_3.get(),
                 b.get(),
                 n,
                 width * 4,
                 height * 4,
                 DECODER_FILTER_3_DEPTH);

    // Copy batch
    memcpy(res.get_data() + offset, b.get(), n * encoded_image_bytes);
    offset += n * encoded_image_bytes;
  }

  // Copy the result
  memcpy(res.get_labels(), dataset.get_labels(), dataset.n * sizeof(int));
  return res;
}

float Cpu_Autoencoder::eval(const Dataset &dataset) {
  return cpu_mse_loss(dataset.get_data(),
                      decode(encode(dataset)).get_data(),
                      dataset.n,
                      dataset.width,
                      dataset.height,
                      dataset.depth);
}

void Cpu_Autoencoder::save_parameters(const char *filename) const {
  ofstream buffer(filename, ios::out | ios::binary);

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

  // Write third decoder conv2D layer
  write_data(buffer, _decoder_filter_3, DECODER_FILTER_3_SIZE * sizeof(float));
  write_data(buffer, _decoder_bias_3, DECODER_FILTER_3_DEPTH * sizeof(float));
}