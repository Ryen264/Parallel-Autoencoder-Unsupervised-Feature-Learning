#include <cstdio>
#include <cstring>
#include <algorithm>

#include "data_loader.h"
#include "cpu_autoencoder.h"

// ============================================================
// Save CIFAR image (CHW layout) to PPM (P6 RGB)
// ============================================================
void save_ppm_chw(const char* filename, float* img) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("❌ Cannot open %s\n", filename);
        return;
    }

    // PPM header
    fprintf(f, "P6\n32 32\n255\n");

    // CIFAR layout: [R(1024) | G(1024) | B(1024)]
    float* R = img;
    float* G = img + 32 * 32;
    float* B = img + 2 * 32 * 32;

    for (int i = 0; i < 32 * 32; i++) {
        unsigned char r = (unsigned char)(std::clamp(R[i], 0.0f, 1.0f) * 255.0f);
        unsigned char g = (unsigned char)(std::clamp(G[i], 0.0f, 1.0f) * 255.0f);
        unsigned char b = (unsigned char)(std::clamp(B[i], 0.0f, 1.0f) * 255.0f);

        fwrite(&r, 1, 1, f);
        fwrite(&g, 1, 1, f);
        fwrite(&b, 1, 1, f);
    }

    fclose(f);
}

// ============================================================
// Main
// ============================================================
int main() {
    // --------------------------------------------------------
    // 1. Load trained autoencoder (WEIGHTS ONLY)
    // --------------------------------------------------------
    Cpu_Autoencoder model;
    
    // Google Colab: use /content/ path
    const char* model_path = "/content/cpu_autoencoder_.bin";
    printf("Loading model from: %s\n", model_path);
    model.load_parameters(model_path);
    printf("✓ Model loaded successfully\n");

    // --------------------------------------------------------
    // 2. Load CIFAR-10 dataset (FLOAT [0,1])
    // --------------------------------------------------------
    // Google Colab: typical CIFAR path
    const char* dataset_dir = "/content/cifar-10-batches-bin";
    printf("Loading dataset from: %s\n", dataset_dir);
    
    // Load all 5 training batches (or adjust n_batches as needed)
    Dataset train = load_dataset(dataset_dir, 5, true);
    printf("✓ Dataset loaded: %d samples\n", train.n);

    // --------------------------------------------------------
    // 3. Take K samples
    // --------------------------------------------------------
    const int K = 8;
    Dataset small(K, train.width, train.height, train.depth);

    // Copy image data
    memcpy(
        small.get_data(),
        train.get_data(),
        K * 32 * 32 * 3 * sizeof(float)
    );
    memcpy(
        small.get_labels(),
        train.get_labels(),
        K * sizeof(int)
    );
    
    printf("✓ Selected %d samples for reconstruction\n", K);

    // --------------------------------------------------------
    // 4. Encode + Decode
    // --------------------------------------------------------
    printf("Encoding and decoding images...\n");
    Dataset recon = model.decode(model.encode(small));

    // Clamp values to [0, 1]
    float* rdata = recon.get_data();
    int total = K * 32 * 32 * 3;
    for (int i = 0; i < total; i++) {
        if (rdata[i] < 0.0f) rdata[i] = 0.0f;
        else if (rdata[i] > 1.0f) rdata[i] = 1.0f;
    }

    // --------------------------------------------------------
    // 5. Debug value range (VERY IMPORTANT)
    // --------------------------------------------------------
    float mn = 1e9f, mx = -1e9f;
    for (int i = 0; i < K * 32 * 32 * 3; i++) {
        mn = std::min(mn, recon.get_data()[i]);
        mx = std::max(mx, recon.get_data()[i]);
    }
    printf("Recon value range: %f -> %f\n", mn, mx);

    // --------------------------------------------------------
    // 6. Save original & reconstructed images
    // --------------------------------------------------------
    // Google Colab: save to /content/ for easy download
    const char* output_dir = "/content/output";
    
    printf("Saving images to %s/...\n", output_dir);
    for (int i = 0; i < K; i++) {
        char orig_name[128];
        char recon_name[128];

        sprintf(orig_name,  "%s/orig_%d.ppm",  output_dir, i);
        sprintf(recon_name, "%s/recon_%d.ppm", output_dir, i);

        save_ppm_chw(orig_name,  small.get_data() + i * 32 * 32 * 3);
        save_ppm_chw(recon_name, recon.get_data() + i * 32 * 32 * 3);
    }

    printf("✅ Saved %d original + reconstructed CIFAR images\n", K);
    printf("Download images from: %s/\n", output_dir);
    
    return 0;
}
