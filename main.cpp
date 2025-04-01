#include <iostream>
#include "dataframe.h"

int main() {

    DataFrame data_frame("test_data_2.csv", true, ",");
    std::cout << data_frame.get_num_features() << std::endl;
    for (auto header : data_frame.head_names) {
        std::cout << header << std::endl;
    }

    auto [train_dataframe, test_dataframe] = data_frame.train_test_split(0.5);

    std::cout << "Train dataframe size: "<< train_dataframe.length() << std::endl;
    std::cout << "Test dataframe size: "<< test_dataframe.length() << std::endl;
    return 0;
}