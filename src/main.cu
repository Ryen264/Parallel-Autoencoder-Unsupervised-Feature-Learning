#include "constants.h"
#include "data_loader.h"
#include "model.h"
#include "visualization.h"
#include "cpu_autoencoder.h"
#include "gpu_autoencoder.h"

#include <iostream>
#include <vector>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <memory>
using namespace std;

string RUN_MODE       = "all";  // "phase_1", "phase_2", "all"
string HARDWARE_MODE  = "gpu";  // "cpu", "gpu"
bool USE_DUMMY_DATA   = false;  // only for phase 2
bool IS_SAVE_MODEL    = true;

const string DATASET_DIR       = "./data/cifar-10-batches-bin";
const string MODEL_OUTPUT_DIR  = "./model";

const string ENCODED_DATASET_FILE = "encoded_dataset.bin";
const string LABELS_FILE          = "labels.bin";
const string SVM_MODEL_FILE       = "svm_model.bin";

const string VISUALIZATION_TRAINING_TIMES_SVG = "training_times.svg";
const string VISUALIZATION_TRAINING_TIMES_CSV = "training_times.csv";
const string VISUALIZATION_SPEEDUP_GRAPH_SVG  = "speedup_graph.svg";
const string VISUALIZATION_SPEEDUP_GRAPH_CSV  = "speedup_data.csv";
const string VISUALIZATION_HTML_DASHBOARD     = "performance_analysis.html";

Dataset dummy_dataset(int n, int width, int height, int depth) {
    unique_ptr<float[]> data(new float[n * width * height * depth]);
    for (int i = 0; i < n * width * height * depth; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return Dataset(data, n, width, height, depth);
}

Dataset phase_1_cpu(const char *dataset_dir, const char *output_dir,
                    bool is_train = true, int n_epoch = 20, int batch_size = 32, float learning_rate = 0.001f, bool verbose = false, int checkpoint = 0) {
    Dataset dataset = load_dataset(dataset_dir, is_train);
    shuffle_dataset(dataset);

    Cpu_Autoencoder autoencoder;
    printf("Training CPU Autoencoder for %d epochs with batch size %d and learning rate %.4f\n", 
           n_epoch, batch_size, learning_rate);
    autoencoder.fit(dataset, n_epoch, batch_size, learning_rate, verbose, checkpoint, output_dir);

    printf("CPU autoencoder MSE = %.4f", autoencoder.eval(dataset));
    return autoencoder.encode(dataset);
}

Dataset phase_1_gpu(const char *dataset_dir, const char *output_dir,
                    bool is_train = true, int n_epoch = 20, int batch_size = 32, float learning_rate = 0.001f, bool verbose = false, int checkpoint = 0) {
    Dataset dataset = load_dataset(dataset_dir, is_train);
    shuffle_dataset(dataset);

    Gpu_Autoencoder autoencoder;
    printf("Training GPU Autoencoder for %d epochs with batch size %d and learning rate %.4f\n", 
           n_epoch, batch_size, learning_rate);
    autoencoder.fit(dataset, n_epoch, batch_size, learning_rate, verbose, checkpoint, output_dir);

    printf("GPU autoencoder MSE = %.4f", autoencoder.eval(dataset));
    return autoencoder.encode(dataset);
}

double phase_2(const Dataset &encoded_dataset, const vector<int> &labels, float train_ratio = 0.8f, bool is_save_model = true) {
    vector<vector<double>> data;
    for (int i = 0; i < encoded_dataset.n; ++i) {
        vector<double> sample(encoded_dataset.width * encoded_dataset.height * encoded_dataset.depth);
        for (int j = 0; j < sample.size(); ++j) {
            sample[j] = encoded_dataset.data[i * sample.size() + j];
        }
        data.push_back(sample);
    }

    // Split into train and test sets
    int train_size = static_cast<int>(train_ratio * encoded_dataset.n);
    vector<vector<double>> trainData(data.begin(), data.begin() + train_size);
    vector<int> trainLabels(labels.begin(), labels.begin() + train_size);
    vector<vector<double>> testData(data.begin() + train_size, data.end());
    vector<int> testLabels(labels.begin() + train_size, labels.end());

    // Train SVM model
    SVMmodel svm_model(C, KERNEL_TYPE, GAMMA_TYPE, TOLERANCE, CACHE_SIZE, MAX_ITER, NOCHANGE_STEPS);
    svm_model.train(trainData, trainLabels);

    // Test SVM model
    vector<int> predictions = svm_model.predict(testData);
    double accuracy = svm_model.calculateAccuracy(testLabels, predictions, NUM_CLASSES);
    vector<double> conf_matrix = svm_model.calculateConfusionMatrix(testLabels, predictions, NUM_CLASSES);

    svm_model.printAccuracy(accuracy);
    svm_model.printConfusionMatrix(conf_matrix);

    if (is_save_model) {
        string svm_model_path = MODEL_OUTPUT_DIR + "/" + SVM_MODEL_FILE;
        svm_model.save(svm_model_path);
    }

    return accuracy;
}

void save_bar_chart(const vector<string>& labels, const vector<double>& times, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return;
    }
    
    file << "Phase,Time(seconds)\n";
    for (size_t i = 0; i < labels.size(); ++i) {
        file << labels[i] << "," << fixed << setprecision(3) << times[i] << "\n";
    }
    file.close();
    cout << "Bar chart data saved to " << filename << endl;
}

void save_speedup_graph(const vector<string>& labels, const vector<double>& speedups, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return;
    }
    
    file << "Phase,Speedup\n";
    for (size_t i = 0; i < labels.size(); ++i) {
        file << labels[i] << "," << fixed << setprecision(3) << speedups[i] << "\n";
    }
    file.close();
    cout << "Speedup graph data saved to " << filename << endl;
}

int main(int argc, char *argv[]) {
    // Override default parameters from command line if provided
    // Usage: ./main [RUN_MODE] [HARDWARE_MODE] [USE_DUMMY_DATA]
    if (argc > 1) {
        RUN_MODE = argv[1];
    }
    if (argc > 2) {
        HARDWARE_MODE = argv[2];
    }
    if (argc > 3) {
        USE_DUMMY_DATA = string(argv[3]) == "true";
    }
    if (argc > 4) {
        IS_SAVE_MODEL = string(argv[4]) == "true";
    }

    // Timing variables
    vector<string> phase_labels;
    vector<double> phase_times;
    double baseline_time = 0.0;
    vector<double> speedups;

    bool RUN_PHASE_1 = (string(RUN_MODE) == "phase_1" || string(RUN_MODE) == "all");
    bool RUN_PHASE_2 = (string(RUN_MODE) == "phase_2" || string(RUN_MODE) == "all");
    if (RUN_PHASE_1) {
        cout << "Phase 1: Training Autoencoder" << endl;
        
        auto start_time = chrono::high_resolution_clock::now();
        bool is_train = true;
        if (HARDWARE_MODE == "gpu") {
            Dataset encoded_dataset = phase_1_gpu(DATASET_DIR.c_str(), MODEL_OUTPUT_DIR.c_str(), is_train,
                                                    N_EPOCH, BATCH_SIZE, LEARNING_RATE, VERBOSE, CHECKPOINT);
        } else {
            Dataset encoded_dataset = phase_1_cpu(DATASET_DIR.c_str(), MODEL_OUTPUT_DIR.c_str(), is_train,
                                                    N_EPOCH, BATCH_SIZE, LEARNING_RATE, VERBOSE, CHECKPOINT);
        }
        auto end_time = chrono::high_resolution_clock::now();
        
        double phase1_time = chrono::duration<double>(end_time - start_time).count();
        phase_labels.push_back("Phase 1 (" + HARDWARE_MODE + " Autoencoder)");
        phase_times.push_back(phase1_time);
        baseline_time = phase1_time;
        speedups.push_back(1.0); // Baseline
        
        cout << "Phase 1 completed in " << fixed << setprecision(2) << phase1_time << " seconds" << endl;
        
        // Save encoded dataset for phase 2
        FILE *f = fopen(ENCODED_DATASET_FILE.c_str(), "wb");
        fwrite(encoded_dataset.data.get(), sizeof(float), encoded_dataset.n * encoded_dataset.width * encoded_dataset.height * encoded_dataset.depth, f);
        fclose(f);
    }

    if (RUN_PHASE_2) {
        cout << "Phase 2: Training SVM on Encoded Data" << endl;
        
        Dataset encoded_dataset;
        vector<int> labels;
        
        if (USE_DUMMY_DATA) {
            // Use dummy data for phase 2
            cout << "Using dummy data for Phase 2" << endl;
            encoded_dataset = dummy_dataset(NUM_TRAIN_SAMPLES, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);
            
            // Create dummy labels
            labels.resize(NUM_TRAIN_SAMPLES);
            for (int i = 0; i < NUM_TRAIN_SAMPLES; ++i) {
                labels[i] = rand() % NUM_CLASSES; // 10 classes
            }
        } else {
            // Load encoded dataset from phase 1
            cout << "Loading encoded dataset from phase 1" << endl;
            FILE *f = fopen(ENCODED_DATASET_FILE.c_str(), "rb");
            if (!f) {
                cerr << "Error: Encoded dataset file not found!" << endl;
                cerr << "To use real data, make sure to run Phase 1 first or set USE_DUMMY_DATA = true" << endl;
                return -1;
            }
            
            int num_samples = NUM_TRAIN_SAMPLES;  // Local variable for actual data size
            unique_ptr<float[]> encoded_data(new float[num_samples * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH]);
            size_t bytes_read = fread(encoded_data.get(), sizeof(float), num_samples * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH, f);
            fclose(f);
            
            if (bytes_read != num_samples * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH) {
                cerr << "Warning: Expected " << (num_samples * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH) 
                         << " elements, but read " << bytes_read << endl;
                num_samples = bytes_read / (IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH);
            }
            
            encoded_dataset = Dataset(encoded_data, num_samples, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH);

            // Load labels from phase 1
            labels.resize(NUM_TRAIN_SAMPLES);
            string labels_path = DATASET_DIR + "/" + LABELS_FILE;
            FILE *lf = fopen(labels_path.c_str(), "rb");
            if (lf) {
                fread(labels.data(), sizeof(int), NUM_TRAIN_SAMPLES, lf);
                fclose(lf);
            } else {
                cerr << "Warning: Labels file not found, using random labels" << endl;
                return -1;
            }
        }

        // Train SVM on encoded data
        auto start_time = chrono::high_resolution_clock::now();
        double accuracy = phase_2(encoded_dataset, labels, TRAIN_RATIO, IS_SAVE_MODEL);
        auto end_time = chrono::high_resolution_clock::now();
        
        double phase2_time = chrono::duration<double>(end_time - start_time).count();
        phase_labels.push_back("Phase 2 (SVM)");
        phase_times.push_back(phase2_time);
        
        // Calculate speedup (if baseline exists)
        if (baseline_time > 0) {
            speedups.push_back(baseline_time / phase2_time);
        } else {
            speedups.push_back(1.0);
        }
        
        cout << "SVM Accuracy on Encoded Data: " << accuracy * 100.0 << "%" << endl;
        cout << "Phase 2 completed in " << fixed << setprecision(2) << phase2_time << " seconds" << endl;
    }

    // Generate visualizations if any phase was run
    if (!phase_times.empty()) {
        cout << "\n========================================" << endl;
        cout << "Performance Analysis" << endl;
        cout << "========================================" << endl;
        
        // Calculate and display total time
        double total_time = 0.0;
        for (double time : phase_times) {
            total_time += time;
        }
        cout << "Total execution time: " << fixed << setprecision(2) << total_time << " seconds" << endl;
        
        // Generate SVG visualizations
        cout << "\nGenerating visualizations..." << endl;
        generate_bar_chart_svg(phase_labels, phase_times, VISUALIZATION_TRAINING_TIMES_SVG);
        save_bar_chart(phase_labels, phase_times, VISUALIZATION_TRAINING_TIMES_CSV);
        
        // Line graph showing cumulative speedup
        if (phase_labels.size() > 1) {
            generate_speedup_graph_svg(phase_labels, speedups, VISUALIZATION_SPEEDUP_GRAPH_SVG);
            save_speedup_graph(phase_labels, speedups, VISUALIZATION_SPEEDUP_GRAPH_CSV);
            
            // Generate HTML dashboard
            generate_visualization_html(VISUALIZATION_TRAINING_TIMES_SVG, VISUALIZATION_SPEEDUP_GRAPH_SVG, VISUALIZATION_HTML_DASHBOARD);
        } else {
            // Generate HTML with only bar chart
            generate_visualization_html(VISUALIZATION_TRAINING_TIMES_SVG, "", VISUALIZATION_HTML_DASHBOARD);
        }
        
        cout << "\n✓ Visualizations generated successfully!" << endl;
        cout << "  - " << VISUALIZATION_TRAINING_TIMES_SVG << " (Bar chart)" << endl;
        if (phase_labels.size() > 1) {
            cout << "  - " << VISUALIZATION_SPEEDUP_GRAPH_SVG << " (Line graph)" << endl;
        }
        cout << "  - " << VISUALIZATION_HTML_DASHBOARD << " (Interactive dashboard)" << endl;
        cout << "\nOpen " << VISUALIZATION_HTML_DASHBOARD << " in your browser to view the charts." << endl;
    }

    return 0;
}