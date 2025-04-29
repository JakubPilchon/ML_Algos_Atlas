#include <iostream>
#include <algorithm>
#include <cmath>

#include "dataframe.h"
#include <numeric>
#include "Models.h"

int main() {

    DataFrame df("generated_dataset.csv", true, ",");

    RegressionTreeModel model(2, 7);

    auto [train_df, test_df] = df.train_test_split(0.2);
    model.fit(train_df);

    double mse =0;

    for (int i =0; i<test_df.length(); i++) {
        auto [data, target] = test_df[i];
        mse += std::pow(target - model.predict(data), 2);
    }

    mse = mse / test_df.length();

    std::cout << "MSE: " << mse << std::endl;

    // under code for classifier tree

    // data_frame.shuffle_data();
    // auto [train_data, test_data] = data_frame.train_test_split(0.1);
    //
    // DecisionTreeModel model(0.9);
    //
    // model.fit(train_data);
    //
    // double good_prediction = 0;
    // for (int i =0; i<test_data.length(); i++) {
    //     auto [data, target] = test_data[i];
    //     double prediction = model.predict(data);
    //     std::cout << "Prediction: " << prediction << std::endl;
    //     std::cout << "True value: " << target << std::endl;
    //
    //     if (prediction == target) {
    //         good_prediction++;
    //     }
    // }
    // std::cout << "Accuracy: " << good_prediction / test_data.length() << std::endl;


    return 0;
}