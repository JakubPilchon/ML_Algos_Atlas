#include <iostream>
#include <algorithm>
#include <numeric>
#include "dataframe.h"
#include "Models.h"

int main() {

    DataFrame data_frame("iris.csv", true, ",");

    auto first_node = std::make_unique<Node>( new Node());
    //
    // first_node->is_leaf = false;
    // first_node->index = 2;
    // first_node->threshold = 0;
    //
    // auto second_node = std::make_unique<Node>(new Node());
    // second_node->is_leaf = true;
    // second_node->value = 1.;
    //
    // auto third_node = std::make_unique<Node>(new Node());
    // third_node->is_leaf = true;
    // third_node->value = 0.;
    //
    //
    // first_node->gretereq = std::move(second_node);
    // first_node->less = std::move(third_node);
    //
    DecisionTreeModel model;
    //
     model.set_first(std::move(first_node));
    // auto [row, target] = data_frame[0];
    // std::cout << model.predict(row) << std::endl;
    // std::cout << "Hello world!" << std::endl;
    model.fit(data_frame);

    std::vector<size_t> row_indexes(data_frame.length());
    std::iota(row_indexes.begin(), row_indexes.end(), 0);

    model.sort_by_feature(row_indexes, 1, data_frame);

    // for (auto i : row_indexes) {
    //     std::cout << data_frame.get_data(i)[1] << std::endl;
    //     std::cout << data_frame.get_target(i) << std::endl;
    // }

    // // find unique categorical elements
    // auto it = std::unique(row_indexes.begin(), row_indexes.end(),
    //     [data_frame](size_t i, size_t j) { return (data_frame.get_target(i) == data_frame.get_target(j)); });
    //
    // for (auto iter = row_indexes.begin(); iter != it; ++iter) {
    //     std::cout << data_frame.get_target(*iter) << " ";
    // }


    return 0;
}