#include <iostream>
#include <vector>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <iomanip>
#include "constants.h"

#include "cpu_autoencoder.h"
#include "data_loader.h"
#include "model.h"
#include "visualization.h"

using namespace std;

string RUN_MODE = "both"; // "phase_1", "phase_2", "both"
bool USE_DUMMY_DATA = false; // only for phase 2

Dataset dummy_dataset(int n, int width, int height, int depth) {
    unique_ptr<float[]> data(new float[n * width * height * depth]);
    for (int i = 0; i < n * width * height * depth; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return Dataset(data, n, width, height, depth);
}

Dataset phase_1_cpu(const char *dataset_dir = "./data/cifar-10-batches-bin", bool is_train = true,
                   int n_epoch = 20, int batch_size = 32, float learning_rate = 0.001f,
                   bool verbose = false, int checkpoint = 0, const char *output_dir = "./model") {
    // Read dataset
    Dataset dataset = load_dataset(dataset_dir, is_train);

    // Shuffle dataset
    shuffle_dataset(dataset);

    // Create and train model
    Cpu_Autoencoder autoencoder;
    autoencoder.fit(dataset, n_epoch, batch_size, learning_rate, verbose, checkpoint, output_dir);

    // Eval
    printf("Autoencoder MSE = %.4f", autoencoder.eval(dataset));
    return autoencoder.encode(dataset);
}

double phase_2(const Dataset &encoded_dataset, const vector<int> &labels, float train_ratio = 0.8f) {
    // Prepare data for SVM
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
    SVMmodel svm_model;
    svm_model.train(trainData, trainLabels);

    // Test SVM model
    double accuracy = svm_model.test(testData, testLabels);
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
    //Override default parameters from command line if provided
    // Usage: ./program [RUN_MODE] [USE_DUMMY_DATA]
    if (argc > 1) {
        RUN_MODE = argv[1];
    }
    if (argc > 2) {
        USE_DUMMY_DATA = string(argv[2]) == "true";
    }

    // Timing variables
    vector<string> phase_labels;
    vector<double> phase_times;
    double baseline_time = 0.0;
    vector<double> speedups;

    bool RUN_PHASE_1 = (string(RUN_MODE) == "phase_1" || string(RUN_MODE) == "both");
    bool RUN_PHASE_2 = (string(RUN_MODE) == "phase_2" || string(RUN_MODE) == "both");
    if (RUN_PHASE_1) {
        cout << "Phase 1: Training Autoencoder" << endl;
        
        auto start_time = chrono::high_resolution_clock::now();
        Dataset encoded_dataset = phase_1_cpu("./data/cifar-10-batches-bin", true);
        auto end_time = chrono::high_resolution_clock::now();
        
        double phase1_time = chrono::duration<double>(end_time - start_time).count();
        phase_labels.push_back("Phase 1 (CPU)");
        phase_times.push_back(phase1_time);
        baseline_time = phase1_time;
        speedups.push_back(1.0); // Baseline
        
        cout << "Phase 1 completed in " << fixed << setprecision(2) 
                  << phase1_time << " seconds" << endl;
        
        // Save encoded dataset for phase 2
        FILE *f = fopen("encoded_dataset.bin", "wb");
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
                labels[i] = rand() % 10; // 10 classes
            }
        } else {
            // Load encoded dataset from phase 1
            cout << "Loading encoded dataset from phase 1" << endl;
            FILE *f = fopen("encoded_dataset.bin", "rb");
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
            FILE *lf = fopen("./data/cifar-10-batches-bin/labels.bin", "rb");
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
        double accuracy = phase_2(encoded_dataset, labels);
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
        cout << "Phase 2 completed in " << fixed << setprecision(2) 
                  << phase2_time << " seconds" << endl;
        
        // Save the trained SVM model
        SVMmodel svm_model;
        svm_model.save("model/svm_model.bin");
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
        cout << "Total execution time: " << fixed << setprecision(2) 
                  << total_time << " seconds" << endl;
        
        // Generate SVG visualizations
        cout << "\nGenerating visualizations..." << endl;
        generate_bar_chart_svg(phase_labels, phase_times, "training_times.svg");
        save_bar_chart(phase_labels, phase_times, "training_times.csv");
        
        // Line graph showing cumulative speedup
        if (phase_labels.size() > 1) {
            generate_speedup_graph_svg(phase_labels, speedups, "speedup_graph.svg");
            save_speedup_graph(phase_labels, speedups, "speedup_data.csv");
            
            // Generate HTML dashboard
            generate_visualization_html("training_times.svg", "speedup_graph.svg", "performance_analysis.html");
        } else {
            // Generate HTML with only bar chart
            generate_visualization_html("training_times.svg", "", "performance_analysis.html");
        }
        
        cout << "\n✓ Visualizations generated successfully!" << endl;
        cout << "  - training_times.svg (Bar chart)" << endl;
        if (phase_labels.size() > 1) {
            cout << "  - speedup_graph.svg (Line graph)" << endl;
        }
        cout << "  - performance_analysis.html (Interactive dashboard)" << endl;
        cout << "\nOpen performance_analysis.html in your browser to view the charts." << endl;
    }

    return 0;
}
