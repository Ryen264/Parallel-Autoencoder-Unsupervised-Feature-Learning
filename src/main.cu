#include "constants.h"
#include "data_loader.h"
#include "cpu_autoencoder.h"
#include "gpu_autoencoder.h"
#include "optimized1_autoencoder.h"
#include "optimized2_autoencoder.h"
#include "optimized_data_loader.h"
#include "model.h"

#include <iostream>
#include <string>
using namespace std;
#include <type_traits>

//./main [version=cpu/gpu/opt1/opt2] [phase_1_mode=train/load] [phase_2_mode=train/load] \
//       [n_batches] [n_epoch] [batch_size] [learning_rate] \
//       [c_param] [kernel_type] [gamma_type]

// Test by just using some first samples
int TRAIN_SAMPLES = -1;
int TEST_SAMPLES  = -1;

// const char *DATASET_DIR          = "./data/cifar-10-batches-bin";
const char *DATASET_DIR = "/content/drive/MyDrive/LapTrinhSongSong/Team Project/data/cifar-10-binary/cifar-10-batches-bin";

// const char *OUTPUT_DIR           = "./output";
const char *OUTPUT_DIR = "/content/output";

// const char *CPU_AUTOENCODER_PATH = "./model/cpu_autoencoder_model.bin";
const char *CPU_AUTOENCODER_PATH = "/content/model/cpu_autoencoder_model.bin";

// const char *GPU_AUTOENCODER_PATH = "./model/gpu_autoencoder_model.bin";
const char *GPU_AUTOENCODER_PATH = "/content/model/gpu_autoencoder_model.bin";

// const char *ENCODED_DATASET_PATH = "./output/encoded_dataset.bin";
const char *ENCODED_DATASET_PATH = "/content/output/encoded_dataset.bin";

// const char *SVM_MODEL_PATH       = "./model/svm_model.bin";
const char *SVM_MODEL_PATH = "/content/model/svm_model.bin";

// const char *SVM_EVAL_PATH        = "./eval/svm_evaluation.txt";
const char *SVM_EVAL_PATH = "/content/eval/svm_evaluation.txt";

// Load and preprocess dataset
Dataset load_dataset_ds(const char *dataset_dir = DATASET_DIR,
                        int         n_batches   = NUM_BATCHES,
                        bool        is_train    = true) {
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
AE phase_1_train(const DT   &dataset,
                 const char *output_dir       = OUTPUT_DIR,
                 const char *autoencoder_path = CPU_AUTOENCODER_PATH,
                 int         n_epoch          = N_EPOCH,
                 int         batch_size       = BATCH_SIZE,
                 float       learning_rate    = LEARNING_RATE,
                 int         checkpoint       = CHECKPOINT,
                 bool        is_save_model    = true) {
  AE autoencoder;
  printf(
      "Training Autoencoder for %d epochs with batch size %d and learning rate %.4f\n",
      n_epoch,
      batch_size,
      learning_rate);
  autoencoder.fit(dataset, n_epoch, batch_size, learning_rate, checkpoint, output_dir);
  printf("Autoencoder Train MSE = %.4f\n", autoencoder.eval(dataset));
  if (is_save_model)
    autoencoder.save_parameters(autoencoder_path);
  return autoencoder;
}

template <typename AE>
AE phase_1_load(const char *autoencoder_path = CPU_AUTOENCODER_PATH) {
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
auto phase_1_encode(const DT   &dataset,
                    const AE   &autoencoder,
                    const char *encoded_dataset_path = ENCODED_DATASET_PATH,
                    bool        is_save_encoded      = true) {
  auto encoded_dataset = autoencoder.encode(dataset);
  printf("Encoded dataset: n=%d, width=%d, height=%d, depth=%d\n",
         encoded_dataset.n,
         encoded_dataset.width,
         encoded_dataset.height,
         encoded_dataset.depth);

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

// Phase 2: Train and evaluate SVM on trainset
template <typename DT>
SVMmodel phase_2_train(const DT   &encoded_dataset,
                       const char *svm_model_path = SVM_MODEL_PATH,
                       float       train_ratio    = TRAIN_RATIO,
                       float       c_param        = C_PARAM,
                       string      kernel_type    = string(KERNEL_PARAM),
                       string      gamma_type     = string(GAMMA_PARAM),
                       float       tolerance      = TOLERANCE,
                       float       cache_size     = CACHE_SIZE,
                       int         max_iter       = MAX_ITER,
                       int         nochange_steps = NOCHANGE_STEPS,
                       int         num_classes    = NUM_CLASSES,
                       bool        is_save_model  = true) {
  vector<vector<double>> data;
  for (int i = 0; i < encoded_dataset.n; ++i) {
    vector<double> sample(
        encoded_dataset.width * encoded_dataset.height * encoded_dataset.depth);
    for (int j = 0; j < sample.size(); ++j) {
      sample[j] = data_ptr(encoded_dataset)[i * sample.size() + j];
    }
    data.push_back(sample);
  }
  vector<int> labels(labels_ptr(encoded_dataset),
                     labels_ptr(encoded_dataset) + encoded_dataset.n);

  // Split into train and test sets
  int train_size = static_cast<int>(train_ratio * encoded_dataset.n);

  vector<vector<double>> trainset(data.begin(), data.begin() + train_size);
  vector<int>            trainLabels(labels.begin(), labels.begin() + train_size);

  vector<vector<double>> validset(data.begin() + train_size, data.end());
  vector<int>            validLabels(labels.begin() + train_size, labels.end());

  // Train SVM model
  SVMmodel svm_model(c_param,
                     kernel_type,
                     gamma_type,
                     tolerance,
                     cache_size,
                     max_iter,
                     nochange_steps);
  svm_model.train(trainset, trainLabels);

  // Test SVM model
  vector<int> predictions = svm_model.predict(validset);
  double accuracy = svm_model.calculateAccuracy(predictions, validLabels, num_classes);
  printf("SVM Train Accuracy (on validation set): %.2f%%\n", accuracy * 100.0);

  // Print classification report
  vector<vector<int>> class_report =
      svm_model.calculateClassificationReport(predictions, validLabels, num_classes);
  svm_model.printClassificationReport(class_report);

  // Print confusion matrix
  vector<vector<int>> conf_matrix =
      svm_model.calculateConfusionMatrix(predictions, validLabels, num_classes);
  svm_model.printConfusionMatrix(conf_matrix);

  if (is_save_model) {
    svm_model.save(svm_model_path);
  }
  return svm_model;
}

// Phase 2: Load SVM model
SVMmodel phase_2_load(const char *svm_model_path = SVM_MODEL_PATH) {
  SVMmodel svm_model;
  svm_model.load(svm_model_path);
  printf("Loaded SVM model from %s\n", svm_model_path);
  return svm_model;
}

// Phase 2: Test SVM on testset
template <typename DT>
double phase_2_test(SVMmodel   &model,
                    const DT   &encoded_dataset,
                    const char *eval_file    = SVM_EVAL_PATH,
                    int         num_classes  = NUM_CLASSES,
                    bool        is_save_eval = true) {
  vector<vector<double>> data;
  for (int i = 0; i < encoded_dataset.n; ++i) {
    vector<double> sample(
        encoded_dataset.width * encoded_dataset.height * encoded_dataset.depth);
    for (int j = 0; j < sample.size(); ++j) {
      sample[j] = data_ptr(encoded_dataset)[i * sample.size() + j];
    }
    data.push_back(sample);
  }
  vector<int> labels(labels_ptr(encoded_dataset),
                     labels_ptr(encoded_dataset) + encoded_dataset.n);

  // Predict using SVM model
  vector<int> predictions = model.predict(data);
  double      accuracy    = model.calculateAccuracy(predictions, labels, num_classes);
  printf("SVM Test Accuracy: %.2f%%\n", accuracy * 100.0);

  // Print classification report
  vector<vector<int>> class_report =
      model.calculateClassificationReport(predictions, labels, num_classes);
  model.printClassificationReport(class_report);

  // Print confusion matrix
  vector<vector<int>> conf_matrix =
      model.calculateConfusionMatrix(predictions, labels, num_classes);
  model.printConfusionMatrix(conf_matrix);

  if (is_save_eval) {
    model.save_evaluation(accuracy, class_report, conf_matrix, eval_file);
  }
  return accuracy;
}

int main(int argc, char *argv[]) {
  string version = "cpu";
  bool train_phase_1 = true;
  bool train_phase_2 = true;

  int         n_batches     = NUM_BATCHES;
  int         n_epoch       = N_EPOCH;
  int         batch_size    = BATCH_SIZE;
  float       learning_rate = LEARNING_RATE;
  float       c_param       = C_PARAM;
  const char *kernel_type   = KERNEL_PARAM;
  const char *gamma_type    = GAMMA_PARAM;

  if (argc > 1)
    version = string(argv[1]);
  if (argc > 2)
    train_phase_1 = (string(argv[2]) == "train") ? true : false;
  if (argc > 3)
    train_phase_2 = (string(argv[3]) == "train") ? true : false;
  if (argc > 4)
    n_batches = atoi(argv[4]);
  if (argc > 5)
    n_epoch = atoi(argv[5]);
  if (argc > 6)
    batch_size = atoi(argv[6]);
  if (argc > 7)
    learning_rate = atof(argv[7]);
  if (argc > 8)
    c_param = atof(argv[8]);
  if (argc > 9)
    kernel_type = argv[9];
  if (argc > 10)
    gamma_type = argv[10];

  cout << "Loading and preprocessing datasets..." << endl;
  Dataset trainset = load_dataset_ds(DATASET_DIR, n_batches, true);
  Dataset testset  = load_dataset_ds(DATASET_DIR, 1, false);

  // Test by just using some first samples
  if (TRAIN_SAMPLES > 0)
    trainset.n = TRAIN_SAMPLES;
  if (TEST_SAMPLES > 0)
    testset.n = TEST_SAMPLES;

  // Branch-specific encoded datasets holders
  Dataset            encoded_trainset_ds, encoded_testset_ds;
  Optimized_Dataset  encoded_trainset_opt, encoded_testset_opt;
  // Phase 1: Train and evaluate Autoencoder on trainset
  if (version == "gpu") {
    Gpu_Autoencoder gpu_autoencoder;
    if (train_phase_1) {
      gpu_autoencoder = phase_1_train<Gpu_Autoencoder, Dataset>(trainset,
                                                       OUTPUT_DIR,
                                                       GPU_AUTOENCODER_PATH,
                                                       n_epoch,
                                                       batch_size,
                                                       learning_rate,
                                                       CHECKPOINT,
                                                       true);
    } else {
      gpu_autoencoder = phase_1_load<Gpu_Autoencoder>(GPU_AUTOENCODER_PATH);
    }
    // Phase 1: Encode trainset and testset
    printf("Encoding trainset and testset using GPU Autoencoder...\n");
    encoded_trainset_ds = phase_1_encode<Gpu_Autoencoder, Dataset>(
        trainset, gpu_autoencoder, ENCODED_DATASET_PATH, true);
    encoded_testset_ds = phase_1_encode<Gpu_Autoencoder, Dataset>(
        testset, gpu_autoencoder, ENCODED_DATASET_PATH, false);
  } else if (version == "cpu") {
    Cpu_Autoencoder cpu_autoencoder;
    if (train_phase_1) {
      cpu_autoencoder = phase_1_train<Cpu_Autoencoder, Dataset>(trainset,
                                                       OUTPUT_DIR,
                                                       CPU_AUTOENCODER_PATH,
                                                       n_epoch,
                                                       batch_size,
                                                       learning_rate,
                                                       CHECKPOINT,
                                                       true);
    } else {
      cpu_autoencoder = phase_1_load<Cpu_Autoencoder>(CPU_AUTOENCODER_PATH);
    }
    // Phase 1: Encode trainset and testset
    printf("Encoding trainset and testset using CPU Autoencoder...\n");
    encoded_trainset_ds = phase_1_encode<Cpu_Autoencoder, Dataset>(
        trainset, cpu_autoencoder, ENCODED_DATASET_PATH, true);
    encoded_testset_ds = phase_1_encode<Cpu_Autoencoder, Dataset>(
        testset, cpu_autoencoder, ENCODED_DATASET_PATH, false);
  } else if (version == "opt1") {
    // Load optimized datasets directly for opt1 path
    Optimized_Dataset opt_trainset = load_dataset_opt(DATASET_DIR, n_batches, true);
    Optimized_Dataset opt_testset  = load_dataset_opt(DATASET_DIR, 1, false);
    Optimized1_Autoencoder opt1_autoencoder;
    if (train_phase_1) {
      opt1_autoencoder = phase_1_train<Optimized1_Autoencoder, Optimized_Dataset>(opt_trainset,
                                                               OUTPUT_DIR,
                                                               GPU_AUTOENCODER_PATH,
                                                               n_epoch,
                                                               batch_size,
                                                               learning_rate,
                                                               CHECKPOINT,
                                                               true);
    } else {
      opt1_autoencoder = phase_1_load<Optimized1_Autoencoder>(GPU_AUTOENCODER_PATH);
    }
    // Phase 1: Encode trainset and testset
    printf("Encoding trainset and testset using Optimized1 Autoencoder...\n");
    encoded_trainset_opt = phase_1_encode<Optimized1_Autoencoder, Optimized_Dataset>(
        opt_trainset, opt1_autoencoder, ENCODED_DATASET_PATH, true);
    encoded_testset_opt = phase_1_encode<Optimized1_Autoencoder, Optimized_Dataset>(
        opt_testset, opt1_autoencoder, ENCODED_DATASET_PATH, false);
  } else if (version == "opt2") {
    // Load optimized datasets directly for opt2 path
    Optimized_Dataset opt_trainset = load_dataset_opt(DATASET_DIR, n_batches, true);
    Optimized_Dataset opt_testset  = load_dataset_opt(DATASET_DIR, 1, false);
    Optimized2_Autoencoder opt2_autoencoder;
    if (train_phase_1) {
      opt2_autoencoder = phase_1_train<Optimized2_Autoencoder, Optimized_Dataset>(opt_trainset,
                                                               OUTPUT_DIR,
                                                               GPU_AUTOENCODER_PATH,
                                                               n_epoch,
                                                               batch_size,
                                                               learning_rate,
                                                               CHECKPOINT,
                                                               true);
    } else {
      opt2_autoencoder = phase_1_load<Optimized2_Autoencoder>(GPU_AUTOENCODER_PATH);
    }
    // Phase 1: Encode trainset and testset
    printf("Encoding trainset and testset using Optimized2 Autoencoder...\n");
    encoded_trainset_opt = phase_1_encode<Optimized2_Autoencoder, Optimized_Dataset>(
        opt_trainset, opt2_autoencoder, ENCODED_DATASET_PATH, true);
    encoded_testset_opt = phase_1_encode<Optimized2_Autoencoder, Optimized_Dataset>(
        opt_testset, opt2_autoencoder, ENCODED_DATASET_PATH, false);

  } else {
    cout << "Invalid version specified. Use 'cpu', 'gpu', or 'opt1'." << endl;
    return -1;
  }
  // // Dummy for testing
  // encoded_trainset = trainset;
  // encoded_testset = testset;

  // Test by just using some first samples
  if (TRAIN_SAMPLES > 0)
    trainset.n = TRAIN_SAMPLES;
  if (TEST_SAMPLES > 0)
    testset.n = TEST_SAMPLES;

  // Phase 2: Train and evaluate SVM on encoded trainset
  SVMmodel svm_model;
  if (train_phase_2) {
    if (version == "opt1") {
      svm_model = phase_2_train(encoded_trainset_opt,
                                SVM_MODEL_PATH,
                                TRAIN_RATIO,
                                c_param,
                                string(kernel_type),
                                string(gamma_type),
                                TOLERANCE,
                                CACHE_SIZE,
                                MAX_ITER,
                                NOCHANGE_STEPS,
                                NUM_CLASSES,
                                true);
    } else {
      svm_model = phase_2_train(encoded_trainset_ds,
                                SVM_MODEL_PATH,
                                TRAIN_RATIO,
                                c_param,
                                string(kernel_type),
                                string(gamma_type),
                                TOLERANCE,
                                CACHE_SIZE,
                                MAX_ITER,
                                NOCHANGE_STEPS,
                                NUM_CLASSES,
                                true);
    }
  } else {
    svm_model = phase_2_load(SVM_MODEL_PATH);
  }

  // Phase 2: Test SVM on encoded testset
  double test_accuracy;
  if (version == "opt1") {
    test_accuracy = phase_2_test(svm_model, encoded_testset_opt, SVM_EVAL_PATH, NUM_CLASSES, true);
  } else {
    test_accuracy = phase_2_test(svm_model, encoded_testset_ds, SVM_EVAL_PATH, NUM_CLASSES, true);
  }
  printf("Final Test Accuracy: %.2f%%\n", test_accuracy * 100.0);

  return 0;
}