//
// Created by kuba on 28.03.25.
//

#ifndef MODELS_H
#define MODELS_H
#include "dataframe.h"
/*
 *  ALL models should inherit from `Model` class.
 *  This way we could use polymorphism i.e. metrics classes
 *  model should implement virtual methods of `Model` class, where:
 *  - `predict` - method for predicting single row
 *  - `fit` method for training the model, this should take as an argument only the Dataframe object reference
 *  every model hyperparameter (i.e. k in KNN) should be taken in constructor, not in `fit` method!
 *  Please also that every model should be written in different files i.e. KNN in knn.cpp, Logistic regression in logres.cpp etc
 *  but headers of the models should be in this file!
 */
class Model {
    public:
        // this method predicts the single target of the row
        virtual double predict(Row) const = 0;

        // this method trains the models
        virtual void fit(const DataFrame&) = 0;
};

class KNNModel : public Model {
    private:
        unsigned int k;

    public:
        explicit KNNModel(unsigned int);
        double predict(Row) const override;
        void fit(const DataFrame&) override;
};

class LinearRegressionModel : public Model {
    private:
        std::vector<double> weights;
        double intercept;
    public:
        explicit LinearRegressionModel();
        double predict(Row) const override;
        std::vector<Row> choose_data(const DataFrame&, int x, int y);
        void fit(const DataFrame&) override;
};
#endif //MODELS_H
