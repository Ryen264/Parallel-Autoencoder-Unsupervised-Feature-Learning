#include "cpu_autoencoder.h"
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
  #include <direct.h>
  #define mkdir(path, mode) _mkdir(path)
#endif

// Khởi tạo trọng số theo He Initialization
void init_weights_he(const unique_ptr<float[]> &arr, int n, int fan_in) {
  float *ptr = arr.get();
  float std_dev = sqrt(2.0f / fan_in);
  
  static default_random_engine generator(time(0));
  normal_distribution<float> distribution(0.0f, std_dev);

  for (int i = 0; i < n; ++i)
    ptr[i] = distribution(generator);
}

// Khởi tạo Bias bằng 0
void init_bias_zero(const unique_ptr<float[]> &arr, int n) {
  float *ptr = arr.get();
  memset(ptr, 0, n * sizeof(float));
}

void read_data(ifstream &buffer, const unique_ptr<float[]> &data, int size) {
  char *ptr = reinterpret_cast<char *>(data.get());
  buffer.read(ptr, size);
}

void write_data(ostream &buffer, const unique_ptr<float[]> &data, int size) {
  const char *ptr = reinterpret_cast<const char *>(data.get());
  buffer.write(ptr, size);
}

Cpu_Autoencoder::Cpu_Autoencoder() {
  _allocate_mem();

  int k_pixels = 9; // 3x3 kernel

  // Layer 1
  init_weights_he(_encoder_filter_1, ENCODER_FILTER_1_SIZE, 3 * k_pixels);
  init_bias_zero(_encoder_bias_1, ENCODER_FILTER_1_DEPTH);

  // Layer 2
  init_weights_he(_encoder_filter_2, ENCODER_FILTER_2_SIZE, ENCODER_FILTER_1_DEPTH * k_pixels);
  init_bias_zero(_encoder_bias_2, ENCODER_FILTER_2_DEPTH);

  // Decoder 1
  init_weights_he(_decoder_filter_1, DECODER_FILTER_1_SIZE, ENCODER_FILTER_2_DEPTH * k_pixels);
  init_bias_zero(_decoder_bias_1, DECODER_FILTER_1_DEPTH);

  // Decoder 2
  init_weights_he(_decoder_filter_2, DECODER_FILTER_2_SIZE, DECODER_FILTER_1_DEPTH * k_pixels);
  init_bias_zero(_decoder_bias_2, DECODER_FILTER_2_DEPTH);

  // Decoder 3
  init_weights_he(_decoder_filter_3, DECODER_FILTER_3_SIZE, DECODER_FILTER_2_DEPTH * k_pixels);
  init_bias_zero(_decoder_bias_3, DECODER_FILTER_3_DEPTH);
};

Cpu_Autoencoder::Cpu_Autoencoder(const char *filename) {
  _allocate_mem();

  ifstream buffer(filename, ios::in | ios::binary);
  if (!buffer.is_open()) {
      printf("Error: Cannot open file %s\n", filename);
      return;
  }

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
};

Cpu_Autoencoder::Cpu_Autoencoder(const Cpu_Autoencoder &other) {
  // Deep copy parameters
  _encoder_filter_1 = make_unique<float[]>(ENCODER_FILTER_1_SIZE);
  memcpy(_encoder_filter_1.get(), other._encoder_filter_1.get(), ENCODER_FILTER_1_SIZE * sizeof(float));
  _encoder_bias_1 = make_unique<float[]>(ENCODER_FILTER_1_DEPTH);
  memcpy(_encoder_bias_1.get(), other._encoder_bias_1.get(), ENCODER_FILTER_1_DEPTH * sizeof(float));

  _encoder_filter_2 = make_unique<float[]>(ENCODER_FILTER_2_SIZE);
  memcpy(_encoder_filter_2.get(), other._encoder_filter_2.get(), ENCODER_FILTER_2_SIZE * sizeof(float));
  _encoder_bias_2 = make_unique<float[]>(ENCODER_FILTER_2_DEPTH);
  memcpy(_encoder_bias_2.get(), other._encoder_bias_2.get(), ENCODER_FILTER_2_DEPTH * sizeof(float));

  _decoder_filter_1 = make_unique<float[]>(DECODER_FILTER_1_SIZE);
  memcpy(_decoder_filter_1.get(), other._decoder_filter_1.get(), DECODER_FILTER_1_SIZE * sizeof(float));
  _decoder_bias_1 = make_unique<float[]>(DECODER_FILTER_1_DEPTH);
  memcpy(_decoder_bias_1.get(), other._decoder_bias_1.get(), DECODER_FILTER_1_DEPTH * sizeof(float));

  _decoder_filter_2 = make_unique<float[]>(DECODER_FILTER_2_SIZE);
  memcpy(_decoder_filter_2.get(), other._decoder_filter_2.get(), DECODER_FILTER_2_SIZE * sizeof(float));
  _decoder_bias_2 = make_unique<float[]>(DECODER_FILTER_2_DEPTH);
  memcpy(_decoder_bias_2.get(), other._decoder_bias_2.get(), DECODER_FILTER_2_DEPTH * sizeof(float));

  _decoder_filter_3 = make_unique<float[]>(DECODER_FILTER_3_SIZE);
  memcpy(_decoder_filter_3.get(), other._decoder_filter_3.get(), DECODER_FILTER_3_SIZE * sizeof(float));
  _decoder_bias_3 = make_unique<float[]>(DECODER_FILTER_3_DEPTH);
  memcpy(_decoder_bias_3.get(), other._decoder_bias_3.get(), DECODER_FILTER_3_DEPTH * sizeof(float));

  // Transient buffers remain null (will be allocated on demand)
}

Cpu_Autoencoder::Cpu_Autoencoder(Cpu_Autoencoder &&other) noexcept
    : _encoder_filter_1(std::move(other._encoder_filter_1)),
      _encoder_bias_1(std::move(other._encoder_bias_1)),
      _encoder_filter_2(std::move(other._encoder_filter_2)),
      _encoder_bias_2(std::move(other._encoder_bias_2)),
      _decoder_filter_1(std::move(other._decoder_filter_1)),
      _decoder_bias_1(std::move(other._decoder_bias_1)),
      _decoder_filter_2(std::move(other._decoder_filter_2)),
      _decoder_bias_2(std::move(other._decoder_bias_2)),
      _decoder_filter_3(std::move(other._decoder_filter_3)),
      _decoder_bias_3(std::move(other._decoder_bias_3)),
      _out_encoder_filter_1(std::move(other._out_encoder_filter_1)),
      _out_encoder_bias_1(std::move(other._out_encoder_bias_1)),
      _out_encoder_relu_1(std::move(other._out_encoder_relu_1)),
      _out_max_pooling_1(std::move(other._out_max_pooling_1)),
      _out_encoder_filter_2(std::move(other._out_encoder_filter_2)),
      _out_encoder_bias_2(std::move(other._out_encoder_bias_2)),
      _out_encoder_relu_2(std::move(other._out_encoder_relu_2)),
      _out_max_pooling_2(std::move(other._out_max_pooling_2)),
      _out_decoder_filter_1(std::move(other._out_decoder_filter_1)),
      _out_decoder_bias_1(std::move(other._out_decoder_bias_1)),
      _out_decoder_relu_1(std::move(other._out_decoder_relu_1)),
      _out_upsampling_1(std::move(other._out_upsampling_1)),
      _out_decoder_filter_2(std::move(other._out_decoder_filter_2)),
      _out_decoder_bias_2(std::move(other._out_decoder_bias_2)),
      _out_decoder_relu_2(std::move(other._out_decoder_relu_2)),
      _out_upsampling_2(std::move(other._out_upsampling_2)),
      _out_decoder_filter_3(std::move(other._out_decoder_filter_3)),
      _out_decoder_bias_3(std::move(other._out_decoder_bias_3)),
      _d_in(std::move(other._d_in)),
      _d_out(std::move(other._d_out)),
      _d_filter(std::move(other._d_filter)) {}

Cpu_Autoencoder &Cpu_Autoencoder::operator=(const Cpu_Autoencoder &other) {
  if (this == &other) return *this;

  // Deep copy parameters
  _encoder_filter_1 = make_unique<float[]>(ENCODER_FILTER_1_SIZE);
  memcpy(_encoder_filter_1.get(), other._encoder_filter_1.get(), ENCODER_FILTER_1_SIZE * sizeof(float));
  _encoder_bias_1 = make_unique<float[]>(ENCODER_FILTER_1_DEPTH);
  memcpy(_encoder_bias_1.get(), other._encoder_bias_1.get(), ENCODER_FILTER_1_DEPTH * sizeof(float));

  _encoder_filter_2 = make_unique<float[]>(ENCODER_FILTER_2_SIZE);
  memcpy(_encoder_filter_2.get(), other._encoder_filter_2.get(), ENCODER_FILTER_2_SIZE * sizeof(float));
  _encoder_bias_2 = make_unique<float[]>(ENCODER_FILTER_2_DEPTH);
  memcpy(_encoder_bias_2.get(), other._encoder_bias_2.get(), ENCODER_FILTER_2_DEPTH * sizeof(float));

  _decoder_filter_1 = make_unique<float[]>(DECODER_FILTER_1_SIZE);
  memcpy(_decoder_filter_1.get(), other._decoder_filter_1.get(), DECODER_FILTER_1_SIZE * sizeof(float));
  _decoder_bias_1 = make_unique<float[]>(DECODER_FILTER_1_DEPTH);
  memcpy(_decoder_bias_1.get(), other._decoder_bias_1.get(), DECODER_FILTER_1_DEPTH * sizeof(float));

  _decoder_filter_2 = make_unique<float[]>(DECODER_FILTER_2_SIZE);
  memcpy(_decoder_filter_2.get(), other._decoder_filter_2.get(), DECODER_FILTER_2_SIZE * sizeof(float));
  _decoder_bias_2 = make_unique<float[]>(DECODER_FILTER_2_DEPTH);
  memcpy(_decoder_bias_2.get(), other._decoder_bias_2.get(), DECODER_FILTER_2_DEPTH * sizeof(float));

  _decoder_filter_3 = make_unique<float[]>(DECODER_FILTER_3_SIZE);
  memcpy(_decoder_filter_3.get(), other._decoder_filter_3.get(), DECODER_FILTER_3_SIZE * sizeof(float));
  _decoder_bias_3 = make_unique<float[]>(DECODER_FILTER_3_DEPTH);
  memcpy(_decoder_bias_3.get(), other._decoder_bias_3.get(), DECODER_FILTER_3_DEPTH * sizeof(float));

  // Clear training/output buffers (they are transient)
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
  _d_in.reset();
  _d_out.reset();
  _d_filter.reset();

  return *this;
}

Cpu_Autoencoder &Cpu_Autoencoder::operator=(Cpu_Autoencoder &&other) noexcept {
  if (this == &other) return *this;

  _encoder_filter_1 = std::move(other._encoder_filter_1);
  _encoder_bias_1   = std::move(other._encoder_bias_1);
  _encoder_filter_2 = std::move(other._encoder_filter_2);
  _encoder_bias_2   = std::move(other._encoder_bias_2);
  _decoder_filter_1 = std::move(other._decoder_filter_1);
  _decoder_bias_1   = std::move(other._decoder_bias_1);
  _decoder_filter_2 = std::move(other._decoder_filter_2);
  _decoder_bias_2   = std::move(other._decoder_bias_2);
  _decoder_filter_3 = std::move(other._decoder_filter_3);
  _decoder_bias_3   = std::move(other._decoder_bias_3);

  // Move transient buffers
  _out_encoder_filter_1 = std::move(other._out_encoder_filter_1);
  _out_encoder_bias_1   = std::move(other._out_encoder_bias_1);
  _out_encoder_relu_1   = std::move(other._out_encoder_relu_1);
  _out_max_pooling_1    = std::move(other._out_max_pooling_1);
  _out_encoder_filter_2 = std::move(other._out_encoder_filter_2);
  _out_encoder_bias_2   = std::move(other._out_encoder_bias_2);
  _out_encoder_relu_2   = std::move(other._out_encoder_relu_2);
  _out_max_pooling_2    = std::move(other._out_max_pooling_2);
  _out_decoder_filter_1 = std::move(other._out_decoder_filter_1);
  _out_decoder_bias_1   = std::move(other._out_decoder_bias_1);
  _out_decoder_relu_1   = std::move(other._out_decoder_relu_1);
  _out_upsampling_1     = std::move(other._out_upsampling_1);
  _out_decoder_filter_2 = std::move(other._out_decoder_filter_2);
  _out_decoder_bias_2   = std::move(other._out_decoder_bias_2);
  _out_decoder_relu_2   = std::move(other._out_decoder_relu_2);
  _out_upsampling_2     = std::move(other._out_upsampling_2);
  _out_decoder_filter_3 = std::move(other._out_decoder_filter_3);
  _out_decoder_bias_3   = std::move(other._out_decoder_bias_3);
  _d_in                 = std::move(other._d_in);
  _d_out                = std::move(other._d_out);
  _d_filter             = std::move(other._d_filter);

  return *this;
}

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

void Cpu_Autoencoder::_allocate_output_mem(int n, int width, int height) {
  int n_pixel = n * width * height;
  _out_encoder_filter_1 = make_unique<float[]>(n_pixel * ENCODER_FILTER_1_DEPTH);
  _out_encoder_bias_1   = make_unique<float[]>(n_pixel * ENCODER_FILTER_1_DEPTH);
  _out_encoder_relu_1   = make_unique<float[]>(n_pixel * ENCODER_FILTER_1_DEPTH);
  
  // [FIXED] avg -> max
  _out_max_pooling_1    = make_unique<float[]>(n_pixel * ENCODER_FILTER_1_DEPTH / 4);

  _out_encoder_filter_2 = make_unique<float[]>(n_pixel * ENCODER_FILTER_2_DEPTH / 4);
  _out_encoder_bias_2   = make_unique<float[]>(n_pixel * ENCODER_FILTER_2_DEPTH / 4);
  _out_encoder_relu_2   = make_unique<float[]>(n_pixel * ENCODER_FILTER_2_DEPTH / 4);
  
  // [FIXED] avg -> max
  _out_max_pooling_2    = make_unique<float[]>(n_pixel * ENCODER_FILTER_2_DEPTH / 16);

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
  constexpr int MAX_FILTER_SIZE = *max_element(FILTER_SIZES, FILTER_SIZES + 5);

  _d_in = make_unique<float[]>(n * width * height * MAX_FILTER_DEPTH);
  _d_out    = make_unique<float[]>(n * width * height * MAX_FILTER_DEPTH);
  _d_filter = make_unique<float[]>(MAX_FILTER_SIZE);
}

void Cpu_Autoencoder::_deallocate_output_mem() {
  _out_encoder_filter_1 = 0;
  _out_encoder_bias_1   = 0;
  _out_encoder_relu_1   = 0;
  
  // [FIXED] avg -> max
  _out_max_pooling_1    = 0;

  _out_encoder_filter_2 = 0;
  _out_encoder_bias_2   = 0;
  _out_encoder_relu_2   = 0;
  
  // [FIXED] avg -> max
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

Dataset Cpu_Autoencoder::_encode_save_output(const Dataset &dataset) {
  int n = dataset.n, width = dataset.width, height = dataset.height,
      depth = dataset.depth;

  cpu_conv2D(dataset.get_data(), _encoder_filter_1.get(), _out_encoder_filter_1.get(),
             n, width, height, depth, ENCODER_FILTER_1_DEPTH);

  cpu_add_bias(_out_encoder_filter_1.get(), _encoder_bias_1.get(), _out_encoder_bias_1.get(),
               n, width, height, ENCODER_FILTER_1_DEPTH);

  cpu_relu(_out_encoder_bias_1.get(), _out_encoder_relu_1.get(),
           n, width, height, ENCODER_FILTER_1_DEPTH);

  // [FIXED] avg -> max
  cpu_max_pooling(_out_encoder_relu_1.get(), _out_max_pooling_1.get(),
                  n, width, height, ENCODER_FILTER_1_DEPTH);

  // [FIXED] avg -> max
  cpu_conv2D(_out_max_pooling_1.get(), _encoder_filter_2.get(), _out_encoder_filter_2.get(),
             n, width / 2, height / 2, ENCODER_FILTER_1_DEPTH, ENCODER_FILTER_2_DEPTH);

  cpu_add_bias(_out_encoder_filter_2.get(), _encoder_bias_2.get(), _out_encoder_bias_2.get(),
               n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH);

  cpu_relu(_out_encoder_bias_2.get(), _out_encoder_relu_2.get(),
           n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH);

  // [FIXED] avg -> max
  cpu_max_pooling(_out_encoder_relu_2.get(), _out_max_pooling_2.get(),
                  n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH);

  Dataset res(n, width / 4, height / 4, ENCODER_FILTER_2_DEPTH);
  memcpy(res.get_data(), _out_max_pooling_2.get(),
         n * (width / 4) * (height / 4) * ENCODER_FILTER_2_DEPTH * sizeof(float));

  return res;
}

Dataset Cpu_Autoencoder::_decode_save_output(const Dataset &dataset) {
  int n = dataset.n, width = dataset.width, height = dataset.height,
      depth = dataset.depth;

  cpu_conv2D(dataset.get_data(), _decoder_filter_1.get(), _out_decoder_filter_1.get(),
             n, width, height, depth, DECODER_FILTER_1_DEPTH);

  cpu_add_bias(_out_decoder_filter_1.get(), _decoder_bias_1.get(), _out_decoder_bias_1.get(),
               n, width, height, DECODER_FILTER_1_DEPTH);

  cpu_relu(_out_decoder_bias_1.get(), _out_decoder_relu_1.get(),
           n, width, height, DECODER_FILTER_1_DEPTH);

  cpu_upsampling(_out_decoder_relu_1.get(), _out_upsampling_1.get(),
                 n, width, height, DECODER_FILTER_1_DEPTH);

  cpu_conv2D(_out_upsampling_1.get(), _decoder_filter_2.get(), _out_decoder_filter_2.get(),
             n, 2 * width, 2 * height, DECODER_FILTER_1_DEPTH, DECODER_FILTER_2_DEPTH);

  cpu_add_bias(_out_decoder_filter_2.get(), _decoder_bias_2.get(), _out_decoder_bias_2.get(),
               n, 2 * width, 2 * height, DECODER_FILTER_2_DEPTH);

  cpu_relu(_out_decoder_bias_2.get(), _out_decoder_relu_2.get(),
           n, 2 * width, 2 * height, DECODER_FILTER_2_DEPTH);

  cpu_upsampling(_out_decoder_relu_2.get(), _out_upsampling_2.get(),
                 n, 2 * width, 2 * height, DECODER_FILTER_2_DEPTH);

  cpu_conv2D(_out_upsampling_2.get(), _decoder_filter_3.get(), _out_decoder_filter_3.get(),
             n, 4 * width, 4 * height, DECODER_FILTER_2_DEPTH, DECODER_FILTER_3_DEPTH);

  cpu_add_bias(_out_decoder_filter_3.get(), _decoder_bias_3.get(), _out_decoder_bias_3.get(),
               n, 4 * width, 4 * height, DECODER_FILTER_3_DEPTH);

  Dataset res(n, 4 * width, 4 * height, DECODER_FILTER_3_DEPTH);
  memcpy(res.get_data(), _out_decoder_bias_3.get(),
         n * 4 * width * 4 * height * DECODER_FILTER_3_DEPTH * sizeof(float));
  return res;
}

float Cpu_Autoencoder::_fit_batch(const Dataset &batch, float learning_rate) {
  int     n = batch.n, width = batch.width, height = batch.height, depth = batch.depth;
  float  *d_in = _d_in.get(), *d_out = _d_out.get(), *d_filter = _d_filter.get();
  Dataset res = _decode_save_output(_encode_save_output(batch));

  float loss = cpu_mse_loss(batch.get_data(), res.get_data(), n, width, height, depth);
  cpu_mse_grad(batch.get_data(), res.get_data(), d_out, n, width, height, depth);

  // --- BACKWARD PASS ---

  // Layer Dec 3 (Conv)
  cpu_bias_grad(d_out, d_in, n, width, height, DECODER_FILTER_3_DEPTH);
  cpu_update_weight(_decoder_bias_3.get(), d_in, DECODER_FILTER_3_DEPTH, learning_rate);

  cpu_conv2D_grad(_out_upsampling_2.get(), d_out, d_filter, n, width, height, DECODER_FILTER_2_DEPTH, DECODER_FILTER_3_DEPTH);
  
  cpu_conv2D_backward_input(d_out, _decoder_filter_3.get(), d_in, n, width, height, DECODER_FILTER_2_DEPTH, DECODER_FILTER_3_DEPTH);
  
  swap(d_out, d_in);
  cpu_update_weight(_decoder_filter_3.get(), d_filter, DECODER_FILTER_3_SIZE, learning_rate);

  // Layer Dec 2 (Upsample -> ReLU -> Conv)
  cpu_upsampling_backward(d_out, d_in, n, width / 2, height / 2, DECODER_FILTER_2_DEPTH);
  cpu_relu_backward(_out_decoder_bias_2.get(), d_in, d_out, n, width / 2, height / 2, DECODER_FILTER_2_DEPTH);

  cpu_bias_grad(d_out, d_in, n, width / 2, height / 2, DECODER_FILTER_2_DEPTH);
  cpu_update_weight(_decoder_bias_2.get(), d_in, DECODER_FILTER_2_DEPTH, learning_rate);

  cpu_conv2D_grad(_out_upsampling_1.get(), d_out, d_filter, n, width / 2, height / 2, DECODER_FILTER_1_DEPTH, DECODER_FILTER_2_DEPTH);
  
  cpu_conv2D_backward_input(d_out, _decoder_filter_2.get(), d_in, n, width / 2, height / 2, DECODER_FILTER_1_DEPTH, DECODER_FILTER_2_DEPTH);
  
  swap(d_out, d_in);
  cpu_update_weight(_decoder_filter_2.get(), d_filter, DECODER_FILTER_2_SIZE, learning_rate);

  // Layer Dec 1 (Upsample -> ReLU -> Conv)
  cpu_upsampling_backward(d_out, d_in, n, width / 4, height / 4, DECODER_FILTER_1_DEPTH);
  cpu_relu_backward(_out_decoder_bias_1.get(), d_in, d_out, n, width / 4, height / 4, DECODER_FILTER_1_DEPTH);

  cpu_bias_grad(d_out, d_in, n, width / 4, height / 4, DECODER_FILTER_1_DEPTH);
  cpu_update_weight(_decoder_bias_1.get(), d_in, DECODER_FILTER_1_DEPTH, learning_rate);

  // [FIXED] avg -> max
  cpu_conv2D_grad(_out_max_pooling_2.get(), d_out, d_filter, n, width / 4, height / 4, ENCODER_FILTER_2_DEPTH, DECODER_FILTER_1_DEPTH);
  
  cpu_conv2D_backward_input(d_out, _decoder_filter_1.get(), d_in, n, width / 4, height / 4, ENCODER_FILTER_2_DEPTH, DECODER_FILTER_1_DEPTH);
  
  swap(d_out, d_in);
  cpu_update_weight(_decoder_filter_1.get(), d_filter, DECODER_FILTER_1_SIZE, learning_rate);

  // Layer Enc 2 (Max Pooling -> ReLU -> Conv)
  cpu_max_pooling_backward(_out_encoder_relu_2.get(), d_out, d_in, n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH);
  
  cpu_relu_backward(_out_encoder_bias_2.get(), d_in, d_out, n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH);

  cpu_bias_grad(d_out, d_in, n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH);
  cpu_update_weight(_encoder_bias_2.get(), d_in, ENCODER_FILTER_2_DEPTH, learning_rate);

  // [FIXED] avg -> max
  cpu_conv2D_grad(_out_max_pooling_1.get(), d_out, d_filter, n, width / 2, height / 2, ENCODER_FILTER_1_DEPTH, ENCODER_FILTER_2_DEPTH);
  
  cpu_conv2D_backward_input(d_out, _encoder_filter_2.get(), d_in, n, width / 2, height / 2, ENCODER_FILTER_1_DEPTH, ENCODER_FILTER_2_DEPTH);
  
  swap(d_out, d_in);
  cpu_update_weight(_encoder_filter_2.get(), d_filter, ENCODER_FILTER_2_SIZE, learning_rate);

  // Layer Enc 1 (Max Pooling -> ReLU -> Conv)
  cpu_max_pooling_backward(_out_encoder_relu_1.get(), d_out, d_in, n, width, height, ENCODER_FILTER_1_DEPTH);
  
  cpu_relu_backward(_out_encoder_bias_1.get(), d_in, d_out, n, width, height, ENCODER_FILTER_1_DEPTH);

  cpu_bias_grad(d_out, d_in, n, width, height, ENCODER_FILTER_1_DEPTH);
  cpu_update_weight(_encoder_bias_1.get(), d_in, ENCODER_FILTER_1_DEPTH, learning_rate);

  cpu_conv2D_grad(batch.get_data(), d_out, d_filter, n, width, height, depth, ENCODER_FILTER_1_DEPTH);
  
  cpu_conv2D_backward_input(d_out, _encoder_filter_1.get(), d_in, n, width, height, depth, ENCODER_FILTER_1_DEPTH);
  
  swap(d_out, d_in);
  cpu_update_weight(_encoder_filter_1.get(), d_filter, ENCODER_FILTER_1_SIZE, learning_rate);

  return loss;
}

void Cpu_Autoencoder::fit(const Dataset &dataset, int n_epoch, int batch_size, float learning_rate, bool verbose, int checkpoint, const char *output_dir) {
  vector<Dataset> batches = create_minibatches(dataset, batch_size);
  _allocate_output_mem(batch_size, dataset.width, dataset.height);

  Timer timer;
  float total_time = 0;

  printf("Training CPU Autoencoder for %d epochs with batch size %d and learning rate %.4f\n", n_epoch, batch_size, learning_rate);
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

      printf(" - Loss = %.4f - Time = %s", total_loss / total, format_time(epoch_time).c_str());
      fflush(stdout);
    }
    total_time += epoch_time;
    puts("\n");

    if (checkpoint > 0 && epoch % checkpoint == 0) {
      stringstream builder;
      builder << output_dir << '/' << "cpu_autoencoder_" << epoch << ".bin";
      save_parameters(builder.str().c_str());
    }
  }
  puts("========================TRAINING END========================");

  _deallocate_output_mem();

  stringstream builder;
  builder << output_dir << '/' << "cpu_autoencoder_.bin";
  save_parameters(builder.str().c_str());

  printf("\nTotal time: %s (ms), Loss: %.4f\n", format_time(total_time).c_str(), eval(dataset));
}

Dataset Cpu_Autoencoder::encode(const Dataset &dataset) const {
  int width = dataset.width, height = dataset.height, depth = dataset.depth;
  int encoded_image_bytes = (width / 4) * (height / 4) * ENCODER_FILTER_2_DEPTH * sizeof(float);
  int offset = 0;

  vector<Dataset> batches = create_minibatches(dataset, ENCODE_BATCH_SIZE);
  Dataset         res(dataset.n, width / 4, height / 4, ENCODER_FILTER_2_DEPTH);

  unique_ptr<float[]> a = make_unique<float[]>(ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH);
  unique_ptr<float[]> b = make_unique<float[]>(ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH);

  for (int i = 0; i < batches.size(); ++i) {
    int n = batches[i].n;

    cpu_conv2D(batches[i].get_data(), _encoder_filter_1.get(), a.get(), n, width, height, depth, ENCODER_FILTER_1_DEPTH);
    cpu_add_bias(a.get(), _encoder_bias_1.get(), b.get(), n, width, height, ENCODER_FILTER_1_DEPTH);
    cpu_relu(b.get(), a.get(), n, width, height, ENCODER_FILTER_1_DEPTH);
    
    // [FIXED] avg -> max
    cpu_max_pooling(a.get(), b.get(), n, width, height, ENCODER_FILTER_1_DEPTH);

    cpu_conv2D(b.get(), _encoder_filter_2.get(), a.get(), n, width / 2, height / 2, ENCODER_FILTER_1_DEPTH, ENCODER_FILTER_2_DEPTH);
    cpu_add_bias(a.get(), _encoder_bias_2.get(), b.get(), n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH);
    cpu_relu(b.get(), a.get(), n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH);

    // [FIXED] avg -> max
    cpu_max_pooling(a.get(), b.get(), n, width / 2, height / 2, ENCODER_FILTER_2_DEPTH);

    memcpy(res.get_data() + offset, b.get(), n * encoded_image_bytes);
    offset += n * encoded_image_bytes;
  }
  memcpy(res.get_labels(), dataset.get_labels(), dataset.n * sizeof(int));
  return res;
}

Dataset Cpu_Autoencoder::decode(const Dataset &dataset) const {
  int width = dataset.width, height = dataset.height, depth = dataset.depth;
  int encoded_image_bytes = 4 * width * 4 * height * DECODER_FILTER_3_DEPTH * sizeof(float);
  int offset = 0;

  vector<Dataset> batches = create_minibatches(dataset, ENCODE_BATCH_SIZE);
  Dataset res(dataset.n, width * 4, height * 4, DECODER_FILTER_3_DEPTH);

  unique_ptr<float[]> a = make_unique<float[]>(ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH);
  unique_ptr<float[]> b = make_unique<float[]>(ENCODE_BATCH_SIZE * width * height * MAX_FILTER_DEPTH);

  for (int i = 0; i < batches.size(); ++i) {
    int n = batches[i].n;

    cpu_conv2D(batches[i].get_data(), _decoder_filter_1.get(), a.get(), n, width, height, depth, DECODER_FILTER_1_DEPTH);
    cpu_add_bias(a.get(), _decoder_bias_1.get(), b.get(), n, width, height, DECODER_FILTER_1_DEPTH);
    cpu_relu(b.get(), a.get(), n, width, height, DECODER_FILTER_1_DEPTH);
    cpu_upsampling(a.get(), b.get(), n, width, height, DECODER_FILTER_1_DEPTH);

    cpu_conv2D(b.get(), _decoder_filter_2.get(), a.get(), n, width * 2, height * 2, DECODER_FILTER_1_DEPTH, DECODER_FILTER_2_DEPTH);
    cpu_add_bias(a.get(), _decoder_bias_2.get(), b.get(), n, width * 2, height * 2, DECODER_FILTER_2_DEPTH);
    cpu_relu(b.get(), a.get(), n, width * 2, height * 2, DECODER_FILTER_2_DEPTH);
    cpu_upsampling(a.get(), b.get(), n, width * 2, height * 2, DECODER_FILTER_2_DEPTH);

    cpu_conv2D(b.get(), _decoder_filter_3.get(), a.get(), n, width * 4, height * 4, DECODER_FILTER_2_DEPTH, DECODER_FILTER_3_DEPTH);
    cpu_add_bias(a.get(), _decoder_bias_3.get(), b.get(), n, width * 4, height * 4, DECODER_FILTER_3_DEPTH);

    memcpy(res.get_data() + offset, b.get(), n * encoded_image_bytes);
    offset += n * encoded_image_bytes;
  }
  memcpy(res.get_labels(), dataset.get_labels(), dataset.n * sizeof(int));
  return res;
}

float Cpu_Autoencoder::eval(const Dataset &dataset) {
  return cpu_mse_loss(dataset.get_data(), decode(encode(dataset)).get_data(), dataset.n, dataset.width, dataset.height, dataset.depth);
}

void Cpu_Autoencoder::save_parameters(const char *filename) const {
  // Create directory if it doesn't exist
  string filepath_str(filename);
  size_t found = filepath_str.find_last_of("/\\");
  if (found != string::npos) {
    string dir_path = filepath_str.substr(0, found);
    struct stat info;
    if (stat(dir_path.c_str(), &info) != 0) {
      printf("Creating directory: %s\n", dir_path.c_str());
      #ifdef _WIN32
        _mkdir(dir_path.c_str());
      #else
        mkdir(dir_path.c_str(), 0777);
      #endif
    }
  }

  ofstream buffer(filename, ios::out | ios::binary);
  if (!buffer.is_open()) {
      fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
      return;
  }

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
  printf("✓ CPU Autoencoder parameters saved successfully to %s\n", filename);
}

void Cpu_Autoencoder::load_parameters(const char *filename) {
  ifstream buffer(filename, ios::in | ios::binary);
  if (!buffer.is_open()) {
      fprintf(stderr, "Error: Cannot open file %s for reading\n", filename);
      return;
  }

  // Đọc lần lượt các trọng số theo đúng thứ tự đã lưu trong save_parameters
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
  printf("✓ CPU Autoencoder parameters loaded successfully from %s\n", filename);
}