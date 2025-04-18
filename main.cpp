#include <iostream>
#include "dataframe.h"
#include "Models.h"

int main() {

    DataFrame data_frame("test_data_2.csv", true, ",");

    data_frame.shuffle_data();
    auto [train_data, test_data] = data_frame.train_test_split(0.1);

    LogisticRegressionModel model;

    model.fit(train_data);

    double good_prediction = 0;
    for (int i =0; i<test_data.length(); i++) {
        auto [data, target] = test_data[i];
        double prediction = model.predict(data);
        std::cout << "Prediction: " << prediction << std::endl;
        std::cout << "True value: " << target << std::endl;

        if (std::max(prediction-target, target-prediction)<0.05) {
            good_prediction++;
        }
    }
    std::cout << "Accuracy: " << good_prediction / test_data.length() << std::endl;
    return 0;
}
