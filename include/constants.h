#ifndef CONSTANTS_H
#define CONSTANTS_H

constexpr int NUM_TRAIN_SAMPLES = 50000;
constexpr int NUM_TEST_SAMPLES  = 10000;
constexpr int NUM_CLASSES       = 10;
constexpr int NUM_BATCHES       = 5;

constexpr int IMAGE_WIDTH  = 32;
constexpr int IMAGE_HEIGHT = 32;
constexpr int IMAGE_DEPTH  = 3;
constexpr int IMAGE_SIZE   = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH;

constexpr int CONV_FILTER_WIDTH  = 3;
constexpr int CONV_FILTER_HEIGHT = 3;

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

// Max block size
constexpr int MAX_BLOCK_SIZE = 1024;

#endif