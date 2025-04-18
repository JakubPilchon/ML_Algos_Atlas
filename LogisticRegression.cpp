//
// Created by Magda on 18.04.25.
//

#include "Models.h"
#include <bits/stdc++.h>


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
            weights[j] -= learning_rate * dw[j] / n_samples;
        }
        bias -= learning_rate * db / n_samples;

      /* ((this is for degubbing))
          if (epoch % 100 == 0) {
            double cost = compute_cost(df);
            std::cout << "Epoch " << epoch << ", Cost: " << cost << std::endl;
        }
        */


    }

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
    weights.resize(a);
}
