//
// Created by Alan Zbucki on 20/04/2025.
//
#include "Models.h"
#include <iostream>
#include <vector>
#include <cmath>

LinearRegressionModel::LinearRegressionModel() : intercept(0.0) {}

// y = w1*x1 + w2*x2 + ... + wn*xn + b
double LinearRegressionModel::predict(Row row) const {
    double result = intercept;
    for (size_t i = 0; i < weights.size(); i++) {
        result += weights[i] * row[i];
    }
    return result;
}

void LinearRegressionModel::fit(const DataFrame& df) {
    size_t n_samples = df.length();
    size_t n_features = df.get_num_features();

    weights = std::vector<double>(n_features, 0.0);
    double intercept = 0.0;

    std::vector<double> mean_x(n_features, 0.0);
    double mean_y = 0.0;

    for (size_t i = 0; i < n_samples; i++) {
        auto [row, target] = df[i];
        for (size_t j = 0; j < n_features; j++){
            mean_x[j] += row[j];
        }
        mean_y += target;
    }

    for (size_t j = 0; j < n_features; j++) {
        mean_x[j] /= n_samples;
    }

    for (size_t j = 0; j < n_features; j++) {
        double numerator = 0.0;
        double denominator = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            auto [row, target] = df[i];
            numerator += (row[j] - mean_x[j]) * (target - mean_y);
            denominator += (row[j] - mean_x[j]) * (row[j] - mean_x[j]);
        }

        weights[j] = numerator / denominator;
    }

    intercept = mean_y;
    for (size_t j = 0; j < n_features; j++) {
        intercept -= weights[j] * mean_x[j];
    }
}