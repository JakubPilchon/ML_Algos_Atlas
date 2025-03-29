//
// Created by kuba on 28.03.25.
//

#ifndef MODELS_H
#define MODELS_H
#include "dataframe.h"

class Model {
    // All the model classes will be inheriting form this class,
    // this way we could use polymorphism which will be important later
    public:
        // this method predicts the single target of the row
        virtual double predict(double[]) const;

        // this method trains the models
        virtual void fit(const DataFrame&);
};

class KNNModel : public Model {
    private:
        unsigned int k;

    public:
        explicit KNNModel(unsigned int);
        double predict(double[]) const override;
        void fit(const DataFrame&) override;
};

#endif //MODELS_H
