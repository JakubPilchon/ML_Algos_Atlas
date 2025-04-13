//
// Created by kuba on 07.04.25.
//

#include "Models.h"
#include <iostream>


double DecisionTreeModel::predict(std::shared_ptr<double[]> row) const {
    if (!first) {
        throw std::runtime_error("first is nullptr");
    }

    auto node = first.get();
    while (node->is_leaf == false) {
        if (row[node->index] >= node->threshold) {
            node = node->gretereq.get();
        } else {
            node = node->less.get();
        }
    }
    double prediction = node->value;
    node = nullptr;
    return prediction;
}

void DecisionTreeModel::set_first(std::unique_ptr<Node> first_node) {
    first = std::move(first_node);
}

void DecisionTreeModel::fit(const DataFrame& df) {

}
