#include <iostream>
#include "dataframe.h"
#include "Models.h"

int main() {

    DataFrame data_frame("iris.csv", true, ",");
    std::cout << data_frame.get_num_features() << std::endl;
    for (auto header : data_frame.head_names) {
        std::cout << header << std::endl;
    }

    auto [train_dataframe, test_dataframe] = data_frame.train_test_split(0.5);

    std::cout << "Train: "  << std::endl;
    train_dataframe.head();

    std::cout << "Test: "  << std::endl;
    test_dataframe.head();

    auto [row, t] = train_dataframe[0];
    std::cout << row[0] << std::endl;
    std::cout << t << std::endl;

    auto row2 = train_dataframe.get_data(0);
    std::cout << row2[0] << std::endl;

    std::cout << "Test regresji liniowej" << std::endl;
    LinearRegressionModel model;
    model.fit(data_frame);
    std::cout << "Model wytrenowany" << std::endl;
    return 0;
}
