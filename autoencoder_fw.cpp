#include "autoencoder.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>

Autoencoder::Autoencoder(int H_, int W_, int C_in_) {
    H = H_; W = W_; C_in = C_in_;

    // filter channels
    conv1_out = 64;
    conv2_out = 128;
    conv3_out = 256;
    conv4_out = 128;
    conv5_out = C_in;

    // allocate weights + biases
    w1 = new float[CONV_FILTER_SIZE*CONV_FILTER_SIZE*C_in*conv1_out]; b1 = new float[conv1_out];
    w2 = new float[CONV_FILTER_SIZE*CONV_FILTER_SIZE*conv1_out*conv2_out]; b2 = new float[conv2_out];
    w3 = new float[CONV_FILTER_SIZE*CONV_FILTER_SIZE*conv2_out*conv3_out]; b3 = new float[conv3_out];
    w4 = new float[CONV_FILTER_SIZE*CONV_FILTER_SIZE*conv3_out*conv4_out]; b4 = new float[conv4_out];
    w5 = new float[CONV_FILTER_SIZE*CONV_FILTER_SIZE*conv4_out*conv5_out]; b5 = new float[conv5_out];

    // allocate intermediate activations
    x1 = new float[H*W*conv1_out];
    x2 = new float[H*W*conv2_out];
    x3 = new float[H*W*conv3_out];
    x4 = new float[H*W*conv4_out];
    x5 = new float[H*W*conv5_out];

    init_weights();
}

Autoencoder::~Autoencoder() {
    delete[] w1; delete[] b1;
    delete[] w2; delete[] b2;
    delete[] w3; delete[] b3;
    delete[] w4; delete[] b4;
    delete[] w5; delete[] b5;

    delete[] x1; delete[] x2; delete[] x3; delete[] x4; delete[] x5;
}

void Autoencoder::init_weights() {
    srand(time(nullptr));
    auto randf = []() { return ((float)rand()/RAND_MAX - 0.5f) * 0.1f; };
    auto init_array = [&](float *arr, int size) { for(int i=0;i<size;i++) arr[i]=randf(); };

    init_array(w1, CONV_FILTER_SIZE*CONV_FILTER_SIZE*C_in*conv1_out); init_array(b1, conv1_out);
    init_array(w2, CONV_FILTER_SIZE*CONV_FILTER_SIZE*conv1_out*conv2_out); init_array(b2, conv2_out);
    init_array(w3, CONV_FILTER_SIZE*CONV_FILTER_SIZE*conv2_out*conv3_out); init_array(b3, conv3_out);
    init_array(w4, CONV_FILTER_SIZE*CONV_FILTER_SIZE*conv3_out*conv4_out); init_array(b4, conv4_out);
    init_array(w5, CONV_FILTER_SIZE*CONV_FILTER_SIZE*conv4_out*conv5_out); init_array(b5, conv5_out);
}

void Autoencoder::forward(float *input, float *output) {
    // ---------------- Encoder ----------------
    conv2D(input, w1, x1, H, C_in, conv1_out, false);  // CPU
    relu(x1, x1, H, conv1_out, false);
    max_pooling(x1, x2, H, conv1_out, false);

    conv2D(x2, w2, x2, H/2, conv1_out, conv2_out, false);
    relu(x2, x2, H/2, conv2_out, false);
    max_pooling(x2, x3, H/2, conv2_out, false);

    conv2D(x3, w3, x3, H/4, conv2_out, conv3_out, false);
    relu(x3, x3, H/4, conv3_out, false);

    // ---------------- Decoder ----------------
    upsampling(x3, x4, H/4, conv3_out, false);
    conv2D(x4, w4, x4, H/2, conv3_out, conv4_out, false);
    relu(x4, x4, H/2, conv4_out, false);

    upsampling(x4, x5, H/2, conv4_out, false);
    conv2D(x5, w5, output, H, conv4_out, conv5_out, false);
}
