#include "constants.h"
#include "data_loader.h"
#include "cpu_autoencoder.h"
#include "gpu_autoencoder.h"
#include "optimized1_autoencoder.h"
#include "optimized2_autoencoder.h"
#include "optimized_data_loader.h"

#include <iostream>
#include <string>
#include <type_traits>
using namespace std;

//./main [version=cpu/gpu/opt1/opt2] [phase_1_mode=train/load] \
//       [n_batches] [n_epoch] [batch_size] [learning_rate] \

// const char *DATASET_DIR          = "./data/cifar-10-batches-bin";
const char *DATASET_DIR = "/content/drive/MyDrive/@fithcmú/LapTrinhSongSong/data/cifar-10-batches-bin";

// const char *OUTPUT_DIR           = "./output/";
const char *OUTPUT_DIR = "/content/output/";
// const char *FULLTRAIN_ENCODED_DATASET_SURFIX = "_fulltrain_encoded_dataset.bin";
const char *TRAIN_ENCODED_DATASET_SURFIX = "_train_encoded_dataset.bin";
const char *TEST_ENCODED_DATASET_SURFIX  = "_test_encoded_dataset.bin";

// const char *MODEL_DIR = "./model/";
const char *MODEL_DIR = "/content/model/";
const char *AUTOENCODER_SURFIX = "_autoencoder.bin";

// Load and preprocess dataset
Dataset load_dataset_ds(const char *dataset_dir = DATASET_DIR, int n_batches = NUM_BATCHES, bool is_train = true) {
  Dataset dataset = read_dataset(dataset_dir, n_batches, is_train);
  shuffle_dataset(dataset);
  return dataset;
}

// Convert Dataset -> Optimized_Dataset for opt1 path without relying on conflicting loaders
Optimized_Dataset to_optimized(const Dataset &src) {
  Optimized_Dataset dst(src.n, src.width, src.height, src.depth);
  size_t count = static_cast<size_t>(src.n) * src.width * src.height * src.depth;
  memcpy(dst.data, src.get_data(), count * sizeof(float));
  memcpy(dst.labels, src.get_labels(), static_cast<size_t>(src.n) * sizeof(int));
  return dst;
}

// Load and preprocess optimized dataset (opt1)
Optimized_Dataset load_dataset_opt(const char *dataset_dir = DATASET_DIR,
                                   int         n_batches   = NUM_BATCHES,
                                   bool        is_train    = true) {
  Optimized_Dataset dataset = read_optimized_dataset(dataset_dir, n_batches, is_train);
  shuffle_optimized_dataset(dataset);
  return dataset;
}

// Phase 1: Train and evaluate Autoencoder on trainset
template <typename AE, typename DT>
AE phase_1_train(const DT   &dataset, const char *autoencoder_path,
                 const char *output_dir = OUTPUT_DIR,
                 int n_epoch = N_EPOCH, int batch_size = BATCH_SIZE, float learning_rate = LEARNING_RATE, int checkpoint = CHECKPOINT, bool is_save_model = true) {
  AE autoencoder;
  printf("Training Autoencoder for %d epochs with batch size %d and learning rate %.4f\n", n_epoch, batch_size, learning_rate);
  autoencoder.fit(dataset, n_epoch, batch_size, learning_rate, checkpoint, output_dir);
  printf("Autoencoder Train MSE = %.4f\n", autoencoder.eval(dataset));

  if (is_save_model)
    autoencoder.save_parameters(autoencoder_path);
  return autoencoder;
}

template <typename AE>
AE phase_1_load(const char *autoencoder_path) {
  AE autoencoder(autoencoder_path);
  printf("Loaded Autoencoder model from %s\n", autoencoder_path);
  return autoencoder;
}

// Phase 1: Encode dataset using trained Autoencoder
// Helpers to access data/labels across dataset types
inline float *data_ptr(const Dataset &d) { return d.get_data(); }
inline int   *labels_ptr(const Dataset &d) { return d.get_labels(); }
inline float *data_ptr(const Optimized_Dataset &d) { return d.data; }
inline int   *labels_ptr(const Optimized_Dataset &d) { return d.labels; }

template <typename AE, typename DT>
auto phase_1_encode(const DT &dataset, const AE &autoencoder, const char *encoded_dataset_path , bool is_save_encoded = true) {
  auto encoded_dataset = autoencoder.encode(dataset);
  printf("Encoded dataset: n=%d, width=%d, height=%d, depth=%d\n", encoded_dataset.n, encoded_dataset.width, encoded_dataset.height, encoded_dataset.depth);

  if constexpr (std::is_same_v<decltype(encoded_dataset), Dataset>) {
    if (is_save_encoded)
      write_binary(encoded_dataset, encoded_dataset_path);
  } else {
    // No generic binary writer for Optimized_Dataset; skip saving.
    (void)encoded_dataset_path;
    (void)is_save_encoded;
  }
  return encoded_dataset;
}

int main(int argc, char *argv[]) {
  string version = "cpu";
  bool train_phase_1 = true;
  const char* dataset_dir = DATASET_DIR;
  int         n_batches     = NUM_BATCHES;
  int         n_epoch       = N_EPOCH;
  int         batch_size    = BATCH_SIZE;
  float       learning_rate = LEARNING_RATE;

  if (argc > 1)
    version = string(argv[1]);
  if (argc > 2)
    train_phase_1 = (string(argv[2]) == "train") ? true : false;
  if (argc > 3)
    dataset_dir = argv[3];
  if (argc > 4)
    n_batches = atoi(argv[4]);
  if (argc > 5)
    n_epoch = atoi(argv[5]);
  if (argc > 6)
    batch_size = atoi(argv[6]);
  if (argc > 7)
    learning_rate = atof(argv[7]);

  cout << "Loading and preprocessing datasets..." << endl;
  // Dataset fulltrainset = load_dataset_ds(dataset_dir, 5, true);
  Dataset trainset = load_dataset_ds(dataset_dir, n_batches, true);
  Dataset testset  = load_dataset_ds(dataset_dir, 1, false);

  // Branch-specific encoded datasets holders
  Dataset            encoded_trainset_ds, encoded_testset_ds; // encoded_fulltrainset_ds;
  Optimized_Dataset  encoded_trainset_opt, encoded_testset_opt; // encoded_fulltrainset_opt;

  string autoencoder_path = string(MODEL_DIR) + version + string(AUTOENCODER_SURFIX);
  // string fulltrain_encoded_dataset_path = string(OUTPUT_DIR) + version + string(FULLTRAIN_ENCODED_DATASET_SURFIX);
  string train_encoded_dataset_path = string(OUTPUT_DIR) + version + string(TRAIN_ENCODED_DATASET_SURFIX);
  string test_encoded_dataset_path = string(OUTPUT_DIR) + version + string(TEST_ENCODED_DATASET_SURFIX);
  bool is_save_model    = true;
  bool is_save_encoded = true;

  // Phase 1: Train and evaluate Autoencoder on trainset
  if (version == "gpu") {
    Gpu_Autoencoder autoencoder;

    if (train_phase_1) {
      autoencoder = phase_1_train<Gpu_Autoencoder, Dataset>(trainset, autoencoder_path.c_str(),
                                                            OUTPUT_DIR, n_epoch, batch_size, learning_rate, CHECKPOINT, is_save_model);
    } else {
      autoencoder = phase_1_load<Gpu_Autoencoder>(autoencoder_path.c_str());
    }

    // Phase 1: Encode trainset and testset
    printf("Encoding trainset and testset using GPU Autoencoder...\n");
    // encoded_fulltrainset_ds = phase_1_encode<Gpu_Autoencoder, Dataset>(fulltrainset, autoencoder, fulltrain_encoded_dataset_path.c_str(), is_save_encoded);
    encoded_trainset_ds = phase_1_encode<Gpu_Autoencoder, Dataset>(trainset, autoencoder, train_encoded_dataset_path.c_str(), is_save_encoded);
    encoded_testset_ds = phase_1_encode<Gpu_Autoencoder, Dataset>(testset, autoencoder, test_encoded_dataset_path.c_str(), is_save_encoded);
  }
  else if (version == "cpu") {
    Cpu_Autoencoder autoencoder;

    if (train_phase_1) {
      autoencoder = phase_1_train<Cpu_Autoencoder, Dataset>(trainset, autoencoder_path.c_str(),
                                                            OUTPUT_DIR, n_epoch, batch_size, learning_rate, CHECKPOINT, is_save_model);
    } else {
      autoencoder = phase_1_load<Cpu_Autoencoder>(autoencoder_path.c_str());
    }

    // Phase 1: Encode trainset and testset
    printf("Encoding trainset and testset using CPU Autoencoder...\n");
    // encoded_fulltrainset_ds = phase_1_encode<Cpu_Autoencoder, Dataset>(fulltrainset, autoencoder, fulltrain_encoded_dataset_path.c_str(), is_save_encoded);
    encoded_trainset_ds = phase_1_encode<Cpu_Autoencoder, Dataset>(trainset, autoencoder, train_encoded_dataset_path.c_str(), is_save_encoded);
    encoded_testset_ds = phase_1_encode<Cpu_Autoencoder, Dataset>(testset, autoencoder, test_encoded_dataset_path.c_str(), is_save_encoded);
  }
  else if (version == "opt1") {
    // Load optimized datasets directly for opt1 path
    // Optimized_Dataset opt_fulltrainset = load_dataset_opt(DATASET_DIR, 5, true);
    Optimized_Dataset opt_trainset = load_dataset_opt(DATASET_DIR, n_batches, true);
    Optimized_Dataset opt_testset  = load_dataset_opt(DATASET_DIR, 1, false);
    Optimized1_Autoencoder autoencoder;

    if (train_phase_1) {
      autoencoder = phase_1_train<Optimized1_Autoencoder, Optimized_Dataset>(opt_trainset, autoencoder_path.c_str(),
                                                                            OUTPUT_DIR, n_epoch, batch_size, learning_rate, CHECKPOINT, is_save_model);
    } else {
      autoencoder = phase_1_load<Optimized1_Autoencoder>(autoencoder_path.c_str());
    }

    // Phase 1: Encode trainset and testset
    printf("Encoding trainset and testset using Optimized1 Autoencoder...\n");
    // encoded_fulltrainset_opt = phase_1_encode<Optimized1_Autoencoder, Optimized_Dataset>(opt_fulltrainset, autoencoder, fulltrain_encoded_dataset_path.c_str(), is_save_encoded);
    encoded_trainset_opt = phase_1_encode<Optimized1_Autoencoder, Optimized_Dataset>(opt_trainset, autoencoder, train_encoded_dataset_path.c_str(), is_save_encoded);
    encoded_testset_opt = phase_1_encode<Optimized1_Autoencoder, Optimized_Dataset>(opt_testset, autoencoder, test_encoded_dataset_path.c_str(), is_save_encoded);
  }
  else if (version == "opt2") {
    // Load optimized datasets directly for opt2 path
    // Optimized_Dataset opt_fulltrainset = load_dataset_opt(DATASET_DIR, 5, true);
    Optimized_Dataset opt_trainset = load_dataset_opt(DATASET_DIR, n_batches, true);
    Optimized_Dataset opt_testset  = load_dataset_opt(DATASET_DIR, 1, false);
    Optimized2_Autoencoder autoencoder;
    
    if (train_phase_1) {
      autoencoder = phase_1_train<Optimized2_Autoencoder, Optimized_Dataset>(opt_trainset, autoencoder_path.c_str(),
                                                                            OUTPUT_DIR, n_epoch, batch_size, learning_rate, CHECKPOINT, is_save_model);
    } else {
      autoencoder = phase_1_load<Optimized2_Autoencoder>(autoencoder_path.c_str());
    }

    // Phase 1: Encode trainset and testset
    printf("Encoding trainset and testset using Optimized2 Autoencoder...\n");
    // encoded_fulltrainset_opt = phase_1_encode<Optimized2_Autoencoder, Optimized_Dataset>(opt_fulltrainset, autoencoder, fulltrain_encoded_dataset_path.c_str(), is_save_encoded);
    encoded_trainset_opt = phase_1_encode<Optimized2_Autoencoder, Optimized_Dataset>(opt_trainset, autoencoder, train_encoded_dataset_path.c_str(), is_save_encoded);
    encoded_testset_opt = phase_1_encode<Optimized2_Autoencoder, Optimized_Dataset>(opt_testset, autoencoder, test_encoded_dataset_path.c_str(), is_save_encoded);

  } else {
    cout << "Invalid version specified. Use 'cpu', 'gpu', 'opt1', or 'opt2'." << endl;
    return -1;
  }
  return 0;
}