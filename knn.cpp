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

KNNModel::KNNModel(unsigned int k_val) : k(k_val) {}

std::vector<Row> train_data;
std::vector<double> train_target;
unsigned int num_features = 0;

void KNNModel::fit(const DataFrame& df) {
    train_data.clear();
    train_target.clear();

    for (size_t i = 0; i < df.length(); ++i) {
        train_data.push_back(df.get_data(i));
        train_target.push_back(df.get_target(i));
    }
    num_features = df.get_num_features();
}
