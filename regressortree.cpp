//
// Created by kuba on 21.04.25.
//
#include "Models.h"
#include <algorithm>
#include <numeric>
#include <iostream>

double RegressionTreeModel::calculate_gini(std::vector<size_t> indexes, const DataFrame &df) const {
    double lorentz_area = 0; //
    double sum = 0;
    sort(indexes.begin(), indexes.end(),
        [df](size_t i1, size_t i2) {return df.get_target(i1) < df.get_target(i2);});


    double last = 0, dy;
    for (size_t i :indexes) {
        dy = df.get_target(i) - last;
        lorentz_area += (dy/2 + sum);
        last = df.get_target(i);
        sum += df.get_target(i);
    }

    return 1 - 2*lorentz_area/(sum*indexes.size());
}

void RegressionTreeModel::fit(const DataFrame &df) {
    if (first) {
        first.release();
    }

    std::vector<size_t> row_indexes(df.length());
    iota(row_indexes.begin(), row_indexes.end(), 0);


    first = build_node(df, row_indexes);
}

NodePtr RegressionTreeModel::build_node(const DataFrame &df, std::vector<size_t> &row_indexes) const {
    double purity = calculate_gini(row_indexes, df);

    if (purity <= min_purity) {
        NodePtr node = std::make_unique<Node>();
        node->is_leaf = true;
        double mean = 0;

        for (size_t i : row_indexes)
            mean += df.get_target(i);

        mean = mean/row_indexes.size();
        node->value = mean;
        std::cout << "created leaf node: " << mean << " Purity: " << purity<< std::endl;
        return node;
    } else {
        // if the data sample is not pure enough we need to get the best split
        size_t  best_split, best_feature;
        double best_purity = 1, current_purity;
        std::vector<size_t> greater_indexes;

        for (size_t feature = 0; feature < df.get_num_features(); feature++ ) {
            sort_by_feature(row_indexes, feature, df);

            for (size_t i = 0; i < row_indexes.size(); i++) {
                greater_indexes.push_back(row_indexes.back());
                row_indexes.pop_back();

                current_purity = (calculate_gini(greater_indexes, df) + calculate_gini(row_indexes, df))/2;

                if (current_purity < best_purity) {
                    best_purity = current_purity;
                    best_split = i;
                    best_feature = feature;
                }
            }
            std::swap(greater_indexes, row_indexes);
        }
        sort_by_feature(row_indexes, best_feature, df);
        NodePtr node = std::make_unique<Node>();
        node->is_leaf = false;
        node->threshold = row_indexes[best_split];

        std::vector<size_t> lesser(row_indexes.begin(), row_indexes.begin() + best_split),
                            greater(row_indexes.begin() + best_split, row_indexes.end());

        node->gretereq = build_node(df, greater);
        node->less = build_node(df, lesser);
        return node;
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


