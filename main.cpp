#include <iostream>
#include <algorithm>
#include <cmath>

#include "metrics.h"
#include "dataframe.h"
#include "Models.h"

int main() {
    DataFrame data_frame("reg_test.csv", true, ",");

    data_frame.shuffle_data();
    auto [train_data, test_data] = data_frame.train_test_split(0.2);

    LogisticRegressionModel model(0.001, 1000);
    DecisionTreeModel tree_model(0.1);

    model.fit(train_data);
    tree_model.fit(test_data);

    std::cout << "Accuracy of log res: " << metrics::accuracy(&model, test_data) << std::endl;
    std::cout << "Accuracy of tree: " << metrics::accuracy(&tree_model, test_data) << std::endl;
    std::cout << "F1 Score: " << metrics::f1_score(&model, test_data) << std::endl;
    return 0;
}
