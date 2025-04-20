//
// Created by Alan Zbucki on 20/04/2025.
//
#include "Models.h"
#include <iostream>
#include <vector>
#include <cmath>

// y = slope * x + intercept
double LinearRegressionModel::predict(Row row) const {
    return slope * row[0] + intercept;
}

// making new data set with only two coluns to use it in linear regression
std::vector<Row> LinearRegressionModel::choose_data(const DataFrame& df, int _x, int _y) {
    // we create new vector which will be our new two-column data set
    std::vector<Row> result;

    // we check if there is enough colums
    if(df.get_num_features() <= std::max(_x,_y)){
        throw std::out_of_range("Given indice is out of range.");
    }

    for (auto row : df) {
          // we make dynamic array for x and y
            Row new_row(new double[2]);
            new_row[0] = row[_x];
            new_row[1] = row[_y];

            result.push_back(new_row);
        }

    return result;
}

void LinearRegressionModel::fit(const DataFrame& df) {
    // we make our two-collumn data set
    auto data = choose_data(df, x, y);

    double mean_x = 0, mean_y = 0;
    size_t n = data.size();

    // calculating mean of x and y
    for (const auto& row : data) {
        mean_x += row[0];
        mean_y += row[1];
    }

    mean_x /= n;
    mean_y /= n;

    // calculating value of slope
    double numerator = 0;
    double denominator = 0;

    for (const auto& row : data) {
      numerator += (row[0] - mean_x) * (row[1] - mean_y);
      denominator += (row[0] - mean_x) * (row[0] - mean_y);
    }

    slope = numerator / denominator;

    // calculating intercept
    intercept = mean_y / slope * mean_x;
}