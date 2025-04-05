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


    // Private constructor for splitting the dataframe
    DataFrame(std::vector<Row> new_data,
              std::vector<double> new_target,
              const std::vector<std::string>& new_head_names,
              unsigned int num_features):
    data(std::move(new_data)), target(std::move(new_target)), head_names(new_head_names), num_features(num_features) {};

    // shuffling for other data
    static void shuffle_data(std::vector<Row>& data, std::vector<double>& target);

public:
    // vector used for header names
    std::vector<std::string> head_names;
    // Main constructor
    DataFrame(const std::string&, bool, const char*);
    // getter for out data, using overloaded []
    std::pair<Row, double> operator[](size_t) const;
    // getter out of target
    double get_target(size_t i) const;
    // getter out of data
    Row get_data(size_t i) const;
    // public method for shuffling itself
    void shuffle_data();
    // prints row in the console
    void print_row(size_t i) const;
    //prints a couple of rows for debugging
    void head(size_t = 5) const;
    // getters for length and number of features
    size_t length() const;
    size_t get_num_features() const;

    //splits dataset into two - training and test sets
    std::pair<DataFrame, DataFrame> train_test_split(float) const;
};


#endif //DATAFRAME_H
