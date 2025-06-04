#include "metrics.h"

double metrics::accuracy(Model* m, DataFrame &df) {
    size_t correct = 0;

    for (size_t i = 0; i < df.length(); ++i) {
        auto [X, y] = df[i];
        double y_pred = m->predict(X);
        //int prediction = y_pred >= 0.5 ? 1 : 0;

        if (y_pred == static_cast<int>(y)) {
            correct++;
        }
    }

    return static_cast<double>(correct) / df.length();
};

double metrics::f1_score(Model *model, DataFrame &df) {
    // only implemented for binary classification
    size_t tp=0, fp=0, fn = 0;
    double f1 =0;

    for (size_t i=0; i< df.length(); i++) {
        auto [X, y] = df[i];

        double y_pred =  model->predict(X);
        if (y_pred == 1 and y == 1.0)
            tp++;
        else if (y_pred == 1 and y == 0.0)
            fp++;
        else if (y_pred == 0 and y == 1.0)
            fn++;
    }
    f1 = static_cast<double>(2*tp) / static_cast<double>(2 * tp + fp + fn); // from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    return f1;
}

double metrics::recall(Model *model, DataFrame &df) {
    size_t p=0, tp =0;

    for (size_t i=0; i< df.length(); i++) {
        auto [X, y] = df[i];

        double y_pred =  model->predict(X);
        if (y== 1.0) {
            p++;
            if (y_pred == 1) {
                tp++;
            }
        }
    }
    if (p ==0)
        throw std::runtime_error("dataframe has no positives in it");

    return static_cast<double>(tp) / static_cast<double>(p);
}

double metrics::precision(Model *model, DataFrame &df) {
    size_t p=0, tp =0;

    for (size_t i=0; i< df.length(); i++) {
        auto [X, y] = df[i];

        double y_pred =  model->predict(X);
        if (y_pred == 1.0) {
            p++;
            if (y == 1) {
                tp++;
            }
        }
    }
    if (p ==0)
        throw std::runtime_error("dataframe has no positives in it");

    return static_cast<double>(tp) / static_cast<double>(p);
}


