//
// Created by kuba on 28.03.25.
//

#ifndef MODELS_H
#define MODELS_H
#include "dataframe.h"
#include <memory>

// using unique_ptr to not to deal with all the memory shanenigans
#define NodePtr std::unique_ptr<Node>

/*
 *  ALL models should inherit from `Model` class.
 *  This way we could use polymorphism i.e. metrics classes
 *  model should implement virtual methods of `Model` class, where:
 *  - `predict` - method for predicting single row
 *  - `fit` method for training the model, this should take as an argument only the Datframe object reference
 *  every model hyperparameter (i.e. k in KNN) should be taken in constructor, not in `fit` method!
 *  Please also that every model should be written in diffrent files i.e. KNN in knn.cpp, Logistic regression in logres.cpp etc
 *  but headers of the models should be in this file!
 */
class Model {

    public:
        virtual ~Model() = default;
        // this method predicts the single target of the row
        virtual double predict(Row) const=0;

        // this method trains the models
        virtual void fit(const DataFrame&)=0;
};

 class KNNModel : public Model {
     private:
         unsigned int k;

     public:
        explicit KNNModel(unsigned int);
         double predict(Row) const override;
         void fit(const DataFrame&) override;
};

// structure representing each node in the tree (both branches and leafes)
struct Node {
    bool is_leaf;
    double value, threshold;
    size_t index;
    NodePtr gretereq = nullptr;
    NodePtr less = nullptr;
};

class DecisionTreeModel : public Model {
    private:
        int counter = 0;
        double calculate_entropy(std::vector<Row>) const;
        NodePtr first = std::make_unique<Node>();;
    public:
        void set_first(NodePtr); // DEBUG METHOD
        double predict(Row) const override;
        void fit(const DataFrame&) override;
};

#endif //MODELS_H
