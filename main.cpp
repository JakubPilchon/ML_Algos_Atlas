#include <iostream>
#include "dataframe.h"
#include "Models.h"

int main() {

    DataFrame data_frame("test_data_2.csv", true, ",");
    std::cout << data_frame.get_num_features() << std::endl;
    for (auto header : data_frame.head_names) {
        std::cout << header << std::endl;
    }

    auto first_node = std::make_unique<Node>( new Node());

    first_node->is_leaf = false;
    first_node->index = 2;
    first_node->threshold = 0;

    auto second_node = std::make_unique<Node>(new Node());
    second_node->is_leaf = true;
    second_node->value = 1.;

    auto third_node = std::make_unique<Node>(new Node());
    third_node->is_leaf = true;
    third_node->value = 0.;


    first_node->gretereq = std::move(second_node);
    first_node->less = std::move(third_node);

    DecisionTreeModel model;

    model.set_first(std::move(first_node));
    auto [row, target] = data_frame[0];
    std::cout << model.predict(row) << std::endl;
    std::cout << "Hello world!" << std::endl;

}