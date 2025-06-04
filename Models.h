//
// Created by kuba on 28.03.25.
//

#ifndef MODELS_H
#define MODELS_H
//#include <format>

#include "dataframe.h"
#include <map>
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

// structure representing each node in the tree (both branches and leafes)
struct Node {
    bool is_leaf;
    double value =-1, threshold =-1;
    size_t index;
    NodePtr gretereq = nullptr;
    NodePtr less = nullptr;
};

class DecisionTreeModel : public Model {

    double min_purity;
    NodePtr first;

    double calculate_entropy(const std::map<double, size_t>&, size_t) const;
    void sort_by_feature(std::vector<size_t>& indexes,  size_t feature, const DataFrame& df) const;
    NodePtr build_node(const DataFrame&, std::vector<size_t>&s ) const;

    public:
        DecisionTreeModel(double min_purity = 0.1)   {
            if (min_purity <= 0 || min_purity > 1) {
                throw std::invalid_argument("min_purity must be between 0 and 1");
            }
            this->min_purity = min_purity;
        };

        double predict(Row) const override;
        void fit(const DataFrame&) override;
};
class LogisticRegressionModel : public Model {
    private:
    // weights and bias are the parameters of our function
    std::vector <double> weights;
    double bias;
    // learning_rate and epochs are parameters for fitting the model
    double learning_rate=0.001;
    double epochs=1000;
public:
    LogisticRegressionModel(double lr, double e);
    double predict(Row) const override;
    void fit(const DataFrame&) override;
//    double compute_cost(const DataFrame&); // testing whether the function accuracy improved or not; used for debugging
    double logit(double) const; // the logarithmic function used in model
    void weights_resize(unsigned int);

};

class RegressionTreeModel : public Model {
    double min_error_value;
    unsigned int  max_depth;
    NodePtr first;

public:
    RegressionTreeModel(double min_error_value = 0.1, unsigned int max_depth=3) : min_error_value(min_error_value), max_depth(max_depth) {};

    double calculate_mse(std::vector<size_t>, const DataFrame&) const;
    void sort_by_feature(std::vector<size_t>& indexes,  size_t feature, const DataFrame& df) const;
    NodePtr build_node(const DataFrame&, std::vector<size_t>&s , unsigned int depth);

    double predict(Row) const override;
    void fit(const DataFrame&) override;
};

#endif //MODELS_H
