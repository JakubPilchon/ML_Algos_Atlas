//
// Created by Magda on 18.04.25.
//

#include "Models.h"
#include <bits/stdc++.h>

LogisticRegressionModel::LogisticRegressionModel(double lr, double e ) :learning_rate(lr), epochs(e), bias(0.0) {}


double LogisticRegressionModel::predict(std::shared_ptr<double[]> row) const {

    double z=bias;
    for (size_t i=0; i<weights.size(); ++i) {
        z += (row[i])*weights[i];
    }
    return logit(z);

}

void LogisticRegressionModel::fit(const DataFrame& df) {
    weights_resize(df.get_num_features());
    size_t n_samples = df.length();
    size_t n_features = df.get_num_features();
    double lambda = 0.01;  //L2 regularization, added

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<double> dw(n_features, 0.0);
        double db = 0.0;

        for (size_t i = 0; i < n_samples; ++i) {
            auto X=df.get_data(i);
            auto y=df.get_target(i);
            double y_pred = predict(X);
            double error = y_pred - y;

            for (size_t j = 0; j < n_features; ++j) {
                dw[j] += X[j] * error;
            }

            db += error;
        }

        for (size_t j = 0; j < n_features; ++j) {
            weights[j] -= learning_rate * dw[j] / n_samples + lambda * weights[j];  //L2, added
        }
        bias -= learning_rate * db / n_samples;

      /* ((this is for degubbing))
          if (epoch % 100 == 0) {
            double cost = compute_cost(df);
            std::cout << "Epoch " << epoch << ", Cost: " << cost << std::endl;
          } */
    }
}

double LogisticRegressionModel::LogLoss(const DataFrame& df) const {  //added
    double loss = 0.0;
    double lambda = 0.01; 

    for (size_t i = 0; i < df.length(); ++i) {
        auto X = df.get_data(i);
        double y = df.get_target(i);
        double y_pred = predict(X);
        
        y_pred = std::max(1e-10, std::min(1.0 - 1e-10, y_pred));
        loss += - (y * std::log(y_pred) + (1 - y) * std::log(1 - y_pred));
    }

    double reg = 0.0;
    for (double w : weights) {
        reg += w * w;
    }
    
    loss += (lambda / 2.0) * reg;

    return loss / df.length();
}

double LogisticRegressionModel::accuracy(const DataFrame& df) const {  //added
    size_t correct = 0;

    for (size_t i = 0; i < df.length(); ++i) {
        auto X = df.get_data(i);
        double y = df.get_target(i);
        double y_pred = predict(X);
        int prediction = y_pred >= 0.5 ? 1 : 0;

        if (prediction == static_cast<int>(y)) {
            correct++;
        }
    }

    return static_cast<double>(correct) / df.length();
}

/*double LogisticRegressionModel::compute_cost(const DataFrame& df) {
    double cost=0.0;

    for (size_t i=0; i<df.length(); ++i) {
        double y_pred = predict(df.get_data(i));
        y_pred = std::max(1e-10, std::min(y_pred, 1.0 - 1e-10));
        cost += -df.get_target(i)*log(y_pred) - (1 - df.get_target(i))*log(1 - y_pred);
    }
    return cost / df.length();
  }*/

double LogisticRegressionModel::logit(double z) const {
    if (z < -50) return 1e-10;
    if (z > 50) return 1 - 1e-10;
    return 1.0/(1.0+exp(-z));
}
void LogisticRegressionModel::weights_resize(unsigned int a) {
    weights.assign(a, 0.0);
}
