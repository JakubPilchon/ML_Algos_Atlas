//
// Created by kuba on 24.03.25.
//

#include <memory>
#include <utility>
#include <vector>
#include <utility>

#define Row std::shared_ptr<double[]>

#ifndef DATAFRAME_H
#define DATAFRAME_H

class DataFrame {
private:
    // Raw data
    std::vector<Row> data;
    // vector holding target features (features we want to predict), we always assume they're last fin out file
    std::vector<double> target;
    unsigned int num_features = 0;

    // shuffling for other data
    void shuffle_data(std::vector<Row>& data, std::vector<double>& target);

public:
    // vector used for header names
    std::vector<std::string> head_names;

    // Main constructor
    DataFrame(const std::string&, bool, const char*);
    // getter for out data, using overloaded []
    Row operator[](size_t) const;
    // getter out of target
    double get_target(size_t i) const;

    // public method for shuffling itself
    void shuffle_data();

    // prints row in the console
    void print_row(size_t i) const;
    // getters for length and number of features
    size_t length() const;
    size_t get_num_features() const;

    //splits dataset into two - training and test sets
    std::pair<DataFrame&, DataFrame&> train_test_split(float);
};

std::pair<DataFrame&, DataFrame&> DataFrame::train_test_split(float test_size) {
/*
    This method splits out dataset into two:
      * training dataset
      * test dataset
      Then it returns a pair of references to two datasets, to avoid unnecessary copying.
      I think you should write separate private constructor that would tak as an argument a reference to data, and other attributes

    Argument `test_size` A value in the range (0, 1)
      specifying the proportion of the original dataset
      to be included in the test dataset

    IMPORTANT:
        There should be no overlap between test and train dataset.
        Meaning no Row should be both in train and test dataset.
        If that were the case we would have what known as DATA_LEAKAGE (https://www.kaggle.com/code/alexisbcook/data-leakage)
*/
}


#endif //DATAFRAME_H
