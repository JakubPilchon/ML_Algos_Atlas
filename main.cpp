#include <iostream>
#include "dataframe.h"

int main() {

    DataFrame data_frame("test_data.csv", true, ",");
    std::cout << data_frame.get_num_features() << std::endl;
    for (auto header : data_frame.head_names) {
        std::cout << header << std::endl;
    }
    data_frame.print_row(0);
    data_frame.print_row(1);
    data_frame.shuffle_data();
    data_frame.print_row(0);
    data_frame.print_row(1);
    //std::cout << "Działa, jej!!!" << std::endl;
    //std::cout << data_frame[3][0] << std::endl;
    return 0;
}