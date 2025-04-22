//
// Created by kuba on 21.04.25.
//
#include "Models.h"
#include <algorithm>

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

}

double RegressionTreeModel::predict(std::shared_ptr<double[]>) const {

}
