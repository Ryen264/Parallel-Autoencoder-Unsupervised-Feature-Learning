#ifndef CONSTANTS_H
#define CONSTANTS_H

constexpr int NUM_TRAIN_SAMPLES = 50000;
constexpr int NUM_TEST_SAMPLES  = 10000;
constexpr int NUM_CLASSES       = 10;
constexpr int NUM_BATCHES       = 5;

constexpr int IMAGE_WIDTH       = 32;
constexpr int IMAGE_HEIGHT      = 32;
constexpr int IMAGE_DEPTH       = 3;
constexpr int IMAGE_SIZE        = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH;

constexpr int CONV_FILTER_WIDTH     = 3;
constexpr int CONV_FILTER_HEIGHT    = 3;
constexpr int MAX_FILTER_DEPTH  = 256;
constexpr int MAX_IMAGE_SIZE    = IMAGE_WIDTH * IMAGE_HEIGHT * MAX_FILTER_DEPTH;

#endif