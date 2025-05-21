//
// Created by kuba on 21.04.25.
//
#include "Models.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <limits>

double RegressionTreeModel::calculate_mse(std::vector<size_t> indexes, const DataFrame &df) const {

    if (indexes.size() == 0) {
        return std::numeric_limits<double>::infinity();
    }

    double mean=0, mse=0;
    // calculate the mean of the given observations samples
    for (auto i : indexes) {
        mean += df.get_target(i);
    }
    mean /= indexes.size();

    // no we calculate the mea nsquared error between the mean and the real taregt values
    for (auto i : indexes) {
        mse += pow(df.get_target(i) - mean, 2);

    }
    return mse/indexes.size();
}

void RegressionTreeModel::fit(const DataFrame &df) {
    if (first) {
        first.release();
    }

    std::vector<size_t> row_indexes(df.length());
    iota(row_indexes.begin(), row_indexes.end(), 0);


    first = build_node(df, row_indexes, 0);
}

NodePtr RegressionTreeModel::build_node(const DataFrame &df, std::vector<size_t> &row_indexes, unsigned int depth)  {
    double error_value = calculate_mse(row_indexes, df);

    if (error_value <= min_error_value || depth == max_depth) {
        NodePtr node = std::make_unique<Node>();
        node->is_leaf = true;
        double mean = 0;

        for (size_t i : row_indexes)
            mean += df.get_target(i);

        mean = mean/(row_indexes.size());
        node->value = mean;
        return node;
    } else {
        // if the data sample is not pure enough we need to get the best split
        size_t  best_split, best_feature;
        double best_error = 1.7976931348623157E+308, current_error;

        for (size_t f = 0; f < df.get_num_features()-1; f++) {
            sort_by_feature(row_indexes, f, df);

            for (size_t i =0; i<row_indexes.size(); i++) {
                std::vector<size_t> lesser_indexes(row_indexes.begin(), row_indexes.begin() + i),
                    greater_indexes(row_indexes.begin() + i, row_indexes.end());


                current_error = (calculate_mse(lesser_indexes, df)*lesser_indexes.size()
                    + calculate_mse(greater_indexes, df)*greater_indexes.size())/row_indexes.size();

                if (current_error < best_error) {
                    best_error = current_error;
                    best_feature = f;
                    best_split = i;
                }
            }
            sort_by_feature(row_indexes, best_feature, df);

            NodePtr node = std::make_unique<Node>();
            node->is_leaf = false;

            std::vector<size_t> lesser(row_indexes.begin(), row_indexes.begin() + best_split),
                                greater(row_indexes.begin() + best_split, row_indexes.end());

            node->threshold = df.get_data(greater[0])[best_feature];
            node->index = best_feature;
            node->gretereq = build_node(df, greater, depth+1);
            node->less = build_node(df, lesser, depth+1);
            return node;
        }

    }
}


double RegressionTreeModel::predict(Row row) const {
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

void RegressionTreeModel::sort_by_feature(std::vector<size_t>& indexes,
                                        size_t feature,
                                        const DataFrame& df) const {
    sort (indexes.begin(), indexes.end(),
    [feature, df](size_t i, size_t j) {return df.get_data(i)[feature] < df.get_data(j)[feature];}); // lambda functon for comparing rows

}


