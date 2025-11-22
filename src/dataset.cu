#include "dataset.h"

Dataset::Dataset(int n, int width, int depth)
    : data(make_unique<float[]>(n * width * width * depth))
    , labels(make_unique<int[]>(n))
    , n(n)
    , width(width)
    , depth(depth) {};

Dataset::Dataset(unique_ptr<float[]> &data, int n, int width, int depth)
    : data(move(data))
    , labels(make_unique<int[]>(n))
    , n(n)
    , width(width)
    , depth(depth) {};

Dataset::Dataset(
    unique_ptr<float[]> &data, unique_ptr<int[]> &labels, int n, int width, int depth)
    : data(move(data)), labels(move(labels)), n(n), width(width), depth(depth) {};

float *Dataset::get_data() const { return data.get(); }
int   *Dataset::get_labels() const { return labels.get(); }