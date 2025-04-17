//
// Created by kuba on 07.04.25.
//

#include "Models.h"
#include <iostream>
#include <algorithm>
#include <map>
#include <math.h>
#include <numeric>


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

// TO BE DELETED
void DecisionTreeModel::set_first(std::unique_ptr<Node> first_node) {
    first = std::move(first_node);
}

void DecisionTreeModel::sort_by_feature(std::vector<size_t>& indexes,
                                        size_t feature,
                                        const DataFrame& df) const {
    sort (indexes.begin(), indexes.end(),
    [feature, df](size_t i, size_t j) {return df.get_data(i)[feature] > df.get_data(j)[feature];}); // lambda functon for comparing rows
    //sort(data.begin(), data.end(), [index](std::shared_ptr<double[]> a, std::shared_ptr<double[]> b) {return a[index] < b[index];});
}

NodePtr DecisionTreeModel::build_node(const DataFrame& df, std::vector<size_t> indexes) const {

}

double DecisionTreeModel::calculate_entropy(const std::map<double, size_t>& val_counts, size_t size) const {
    double entropy = 0;
    for (auto value : val_counts) {
        if (value.second != 0) {
            double p = static_cast<double>(value.second) / size;
            entropy += -p * std::log2(p);
        }
    }
    return entropy;
}



void DecisionTreeModel::fit(const DataFrame& df) {
    // deletes the first pointer in case model already has been trained
    // effectively reseting it
    if (first) {
        first.release();
    }

    std::vector<size_t> row_indexes(df.length());
    std::iota(row_indexes.begin(), row_indexes.end(), 0);

    // find unique categorical elements
    // auto it = std::unique(row_indexes.begin(), row_indexes.end(),
    //     [df](size_t i, size_t j) { return (df.get_target(i) == df.get_target(j)); });

    std::map<double, size_t> value_counts;

    for (auto index : row_indexes) {
        double val = df.get_target(index);
        if (value_counts.contains(val)) {
            value_counts[val]++;
        } else {
            value_counts.insert(std::make_pair(val, 1));
        }
    }


    for (auto it : value_counts) {

        std::cout << it.first << " " << it.second << std::endl;
    }
    std::cout << "Entropy: " << calculate_entropy(value_counts, row_indexes.size()) <<std::endl;
}
