#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <string>
using namespace std;

constexpr int NUM_BATCHES       = 1;
constexpr int NUM_PER_BATCH     = 10000;
constexpr int NUM_TRAIN_SAMPLES = NUM_BATCHES * NUM_PER_BATCH;
constexpr int NUM_CLASSES       = 10;
constexpr int NUM_TEST_SAMPLES  = 10000;

constexpr int   N_EPOCH       = 20;
constexpr int   BATCH_SIZE    = 32;
constexpr float LEARNING_RATE = 0.001f;
constexpr bool  VERBOSE       = false;
constexpr int   CHECKPOINT    = 0;
constexpr float TRAIN_RATIO   = 0.8f;

constexpr int IMAGE_WIDTH  = 32;
constexpr int IMAGE_HEIGHT = 32;
constexpr int IMAGE_DEPTH  = 3;
constexpr int IMAGE_SIZE   = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH;

// SVM parameters
constexpr float  C_PARAM            = 10.0f;
constexpr const char* KERNEL_PARAM  = "RBF";
constexpr const char* GAMMA_PARAM   = "auto";
constexpr float  TOLERANCE          = 1e-3f;
constexpr float  CACHE_SIZE         = 200.0f; // in MB
constexpr int    MAX_ITER           = 100;
constexpr int    NOCHANGE_STEPS     = 100;
constexpr int    VERBOSITY          = 0;
constexpr int    CONV_FILTER_WIDTH  = 3;
constexpr int    CONV_FILTER_HEIGHT = 3;

constexpr int MAX_FILTER_DEPTH = 256;
constexpr int MAX_IMAGE_SIZE   = IMAGE_WIDTH * IMAGE_HEIGHT * MAX_FILTER_DEPTH;

// The first conv2d layer of the encoder
constexpr int ENCODER_FILTER_1_DEPTH = 256;
constexpr int ENCODER_FILTER_1_SIZE =
    CONV_FILTER_WIDTH * CONV_FILTER_HEIGHT * IMAGE_DEPTH * ENCODER_FILTER_1_DEPTH;

// The second conv2d layer of the encoder
constexpr int ENCODER_FILTER_2_DEPTH = 128;
constexpr int ENCODER_FILTER_2_SIZE  = CONV_FILTER_WIDTH *
                                      CONV_FILTER_HEIGHT *
                                      ENCODER_FILTER_1_DEPTH *
                                      ENCODER_FILTER_2_DEPTH;

// The first conv2d layer of the decoder
constexpr int DECODER_FILTER_1_DEPTH = 128;
constexpr int DECODER_FILTER_1_SIZE  = CONV_FILTER_WIDTH *
                                      CONV_FILTER_HEIGHT *
                                      ENCODER_FILTER_2_DEPTH *
                                      DECODER_FILTER_1_DEPTH;

// The second conv2d layer of the decoder
constexpr int DECODER_FILTER_2_DEPTH = 256;
constexpr int DECODER_FILTER_2_SIZE  = CONV_FILTER_WIDTH *
                                      CONV_FILTER_HEIGHT *
                                      DECODER_FILTER_1_DEPTH *
                                      DECODER_FILTER_2_DEPTH;

// The third conv2d layer of the decoder
constexpr int DECODER_FILTER_3_DEPTH = IMAGE_DEPTH;
constexpr int DECODER_FILTER_3_SIZE  = CONV_FILTER_WIDTH *
                                      CONV_FILTER_HEIGHT *
                                      DECODER_FILTER_2_DEPTH *
                                      DECODER_FILTER_3_DEPTH;

// Encode and decode by batches to use less memory
constexpr int ENCODE_BATCH_SIZE = 1024;

// Block size to use
constexpr int MAX_BLOCK_SIZE = 1024;

#endif