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



void DecisionTreeModel::sort_by_feature(std::vector<size_t>& indexes,
                                        size_t feature,
                                        const DataFrame& df) const {
    sort (indexes.begin(), indexes.end(),
    [feature, df](size_t i, size_t j) {return df.get_data(i)[feature] < df.get_data(j)[feature];}); // lambda functon for comparing rows
}

NodePtr DecisionTreeModel::build_node(const DataFrame& df, std::vector<size_t>& row_indexes) const {
    /*
     * Check purity of dataset
     * if the data is pure enough create leaf node
     * else crate branch node with recursive children building
     */
    double  max_category; // most common category in data
    double purity =0; // number of most common category in data
    std::map<double, size_t> less_value_counts, greater_value_counts; // holds the number of categories in data


    // create value counts
    for (auto index : row_indexes) {
        double val = df.get_target(index);
        if (less_value_counts.contains(val)) {
            less_value_counts[val]++;
        } else {
            less_value_counts.insert(std::make_pair(val, 1));
            greater_value_counts.insert(std::make_pair(val, 0));
        }
    }

    // calculate purity
    for (auto [category, count] : less_value_counts) {
        if (count > purity) {
            purity = static_cast<double>(count);
            max_category = category;
        }
    }
    purity = purity / row_indexes.size();

    if (purity >= min_purity) {
        // Create leaf node if the data is pure enough
        NodePtr node = std::make_unique<Node>();
        node->is_leaf = true;
        node->value = max_category;
        return node;
    } else {
        // if the data is not pure enough, make split (or branch node)
        double category, best_feature, min_entropy = 1.7976931348623157E+308;
        size_t best_split=0, counter;

        // search every feature for best split
        for (size_t feature = 0; feature < df.get_num_features(); feature++ ) {
            counter = 0;
            sort_by_feature(row_indexes, feature, df);

            for (size_t i : row_indexes) {
                counter++;
                category = df.get_target(i);
                less_value_counts[category]--;
                greater_value_counts[category]++;


                double entropy = calculate_entropy(greater_value_counts, counter) +
                                 calculate_entropy(less_value_counts, row_indexes.size() -counter);

                // if the current split is better than the previous ones, save it
                if (entropy < min_entropy) {
                    min_entropy = entropy;
                    best_feature = feature;
                    best_split = counter;
                    }
                }

            // reset the value_counts for the next iterations
            std::swap(less_value_counts, greater_value_counts);
            }

        sort_by_feature(row_indexes, best_feature, df);
        std::vector<size_t> less_indexes(row_indexes.begin(), row_indexes.begin() + best_split),
                    greater_indexes(row_indexes.begin() + best_split, row_indexes.end());

        //
        NodePtr node = std::make_unique<Node>();
        node->is_leaf = false;
        node->index = best_feature;
        node->threshold = df.get_data(greater_indexes[0])[best_feature];

        //recursively build the other nodes
        node->less = build_node(df, less_indexes);
        node->gretereq = build_node(df, greater_indexes);
        return node;
    }

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

    first = build_node(df, row_indexes);
}
