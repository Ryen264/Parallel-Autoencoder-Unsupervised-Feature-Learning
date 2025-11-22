#include <iostream>
#include <vector>
#include <chrono>
#include "cpu_autoencoder.cu"
#include "layers.h"

// Giả sử bạn đã có class Images và data loader từ ./data_loader
#include "data_loader.h"

int main() {
    // Hyperparameters
    const int batch_size = 32;
    const int epochs     = 20;
    const float lr       = 0.001f; // chỉ lưu để backprop sau này

    // Load dataset
    Dataset cifar10;
    cifar10.load_training_data();
    cifar10.load_test_data();

    int n_train = cifar10.train_images().n;
    int n_batches = (n_train + batch_size - 1) / batch_size;

    std::cout << "Starting training...\n";

    Cpu_Autoencoder autoencoder; // CPU autoencoder

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();
        float epoch_loss = 0.0f;

        // Shuffle dataset each epoch
        cifar10.shuffle_training_data();

        for (int b = 0; b < n_batches; ++b) {
            // Lấy batch
            Images batch = cifar10.get_train_batch(batch_size, b);

            // Forward pass
            Images encoded = autoencoder.encode(batch);
            Images decoded = autoencoder.decode(encoded);

            // Compute loss
            float loss = mse_loss(batch.get(), decoded.get(), batch.width, batch.depth, false);
            epoch_loss += loss;

            // TODO: Backward pass + weight update nếu muốn
            // hiện tại chỉ baseline CPU forward + loss
        }

        epoch_loss /= n_batches;

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "] "
                  << "Loss: " << epoch_loss
                  << ", Time: " << elapsed << "s\n";
    }

    // Save trained model parameters
    autoencoder._save_paramters("trained_model.bin");

    std::cout << "Training completed. Model saved as trained_model.bin\n";

    return 0;
}
