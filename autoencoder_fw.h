#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "layers.h"  

class Autoencoder {
public:
    int H, W, C_in;

    // số lượng filter mỗi layer
    int conv1_out, conv2_out, conv3_out, conv4_out, conv5_out;

    // weights và biases
    float *w1, *b1;
    float *w2, *b2;
    float *w3, *b3;
    float *w4, *b4;
    float *w5, *b5;

    // intermediate activations
    float *x1, *x2, *x3, *x4, *x5;

    Autoencoder(int H, int W, int C_in);
    ~Autoencoder();

    void init_weights();
    void forward(float *input, float *output);
};

#endif
