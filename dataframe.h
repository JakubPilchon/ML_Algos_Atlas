//
// Created by kuba on 24.03.25.
//

#include <memory>
#include <vector>

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
public:
    // vector used for header names
    std::vector<std::string> head_names;

    // Main constructor
    DataFrame(const std::string&, bool, const char*);
    // getter for out data, using overloaded []
    Row operator[](size_t) const;
    // getter out of target
    double get_target(size_t i) const;
    // shuffles our data
    void shuffle_data(std::vector<Row>& data, std::vector<double>& target);

    // prints row in the console
    void print_row(size_t i) const;
    // getters for length and number of features
    size_t length() const;
    size_t get_num_features() const;
};


#endif //DATAFRAME_H
