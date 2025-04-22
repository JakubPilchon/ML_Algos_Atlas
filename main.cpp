#include <iostream>
#include <algorithm>
#include "dataframe.h"
#include <numeric>
#include "Models.h"

int main() {

    DataFrame data_frame("gini_coefficient_test_data.csv", true, ",");

    RegressionTreeModel model;

    std::vector<size_t> row_indexes(data_frame.length());
    iota(row_indexes.begin(), row_indexes.end(), 0);

    std::cout << model.calculate_gini(row_indexes ,data_frame) << std::endl;



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