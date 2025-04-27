#include "Models.h"
#include <iostream>
#include <vector>
#include <cmath>


double euclideanDistance(const Row& a, const Row& b, size_t num_features) {
    double sum = 0.0;
    for (size_t i = 0; i < num_features; ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}