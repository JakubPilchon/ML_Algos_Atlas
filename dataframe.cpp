//
// Created by kuba on 24.03.25.
//
#include <fstream>
#include <iostream>
#include <err.h>
#include "dataframe.h"
#include <sstream>
#include <algorithm>
#include <random>
//#include <__random/random_device.h>

DataFrame::DataFrame(const std::string& file_name, bool uses_headers, const char* delimeter = ",") {
    std::ifstream data_file(file_name, std::ios::binary);

    // Try to open file, in other case throw error
    if (!data_file.is_open()) {
     throw std::runtime_error("Could not open file " + file_name);
    }

    if (uses_headers) { // Case if headers are in a file
        std::string headers, temp;
        if (getline(data_file, headers)) {

            for (auto c : headers) {
                if (c != *delimeter) {
                    temp += c;
                } else {
                    head_names.push_back(temp);
                    temp.clear();
                    num_features++;
                }
            }
            head_names.push_back(temp);
            temp.clear();

        } else {
            throw std::runtime_error("File is empty");
        }
    } else { // Case if headers are not provided
        // To Do Later
        // We assume that num features is non-zero
        std::string first_line;
        if (std::getline(data_file, first_line)) {
            std::stringstream ss(first_line); //we make string stream called ss that helps us divide first_line with delimiter
            std::string first_words;
            // we dont know the number of features, so we use vector
            std::vector<double> first_features;

            // we read the numbers from first row into the vector
            while (std::getline(ss, first_words, *delimeter)) { //counting how many columns
                first_features.push_back(std::stod(first_words));
            }

            Row first_row(new double[first_features.size() - 1]);
            for (int i = 0; i < first_features.size() - 1; i++) {
                first_row[i] = first_features[i];
            }

            data.push_back(first_row);
            target.push_back(first_features.back());
            num_features = first_features.size() - 1;

        } else {
            throw std::runtime_error("File is empty");
        }
    }

    // After we read header names and got feature names, we can load out our data
    std::string row_text, number;
    int i = 0;
    while (std::getline(data_file, row_text)) {
        Row row = Row(new double[num_features]);
        for (auto c : row_text) {
            // split the line by out delimiter, then one-by-one add a new number to our row
            if (c != *delimeter) {
                number.push_back(c);
            } else {
                row[i] = std::stod(number);
                i++;
                number.clear();
            }

        }
        // we push the last number into target
        target.push_back(stod(number));
        data.push_back(row);
        i = 0;
        number.clear();
    }
}

Row  DataFrame::operator[](size_t index) const {
    return data[index];
}

void DataFrame::print_row(const size_t i) const{
    if (i >= data.size())
        throw std::out_of_range("Row index out of range");
     std::cout << i << ": ";
     for (uint j = 0; j < num_features; j++) {
        std::cout << data[i][j] << " ";
     }
     std::cout << std::endl;
}

double  DataFrame::get_target(const size_t i) const {
    return target[i];
}

size_t DataFrame::length() const {
    return data.size();
}

size_t DataFrame::get_num_features() const {
    return num_features;
}

void DataFrame::shuffle_data(std::vector<Row>& data, std::vector<double>& target) {
    if (data.size() != target.size()) {
        throw std::out_of_range("Data size mismatch");
    }

    std::vector<size_t> idxs(data.size());
    for (size_t i = 0; i < idxs.size(); i++) { // We make vector with indices {0,...,n-1}
        idxs[i] = i;
    }

    std::random_device rd; // seed
    std::mt19937 gen(rd()); // Generator of pseudo-random numbers

    std::shuffle(idxs.begin(), idxs.end(), gen); // We shuffle our indices

    std::vector<Row> shuffled_data;
    std::vector<double> shuffled_target;

    for (size_t i : idxs) {
        shuffled_data.push_back(data[i]);
        shuffled_target.push_back(target[i]);
    }

    data = std::move(shuffled_data); // We move contents of shuffled_data to data instead of copying it
    target = std::move(shuffled_target);
}


void DataFrame::shuffle_data() {
    DataFrame::shuffle_data(data, target);
}


std::pair<DataFrame, DataFrame> DataFrame::train_test_split(float test_size) {
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

    //Validating test_size
    if (test_size <= 0 || test_size >= 1)
        throw std::out_of_range("Test size must be between 0 and 1");

    //calculate number of observations in test dataset + checks that atleast one row will be in test dataset
    int test_num = (test_size * data.size());
    if (test_num  == 0) {
        throw std::runtime_error("Too small test size for this dataset");
    }

    // copy neccessary data
    std::vector<Row> shuffled_data(data);
    std::vector<double> shuffled_target(target);

    //shuffle data and target
    DataFrame::shuffle_data(shuffled_data, shuffled_target);

    // create train and test subsets
    std::vector<Row> test_data(shuffled_data.begin(), shuffled_data.begin() + test_num),
                     train_data(shuffled_data.begin() + test_num, shuffled_data.end());


    std::vector<double> test_target(shuffled_target.begin(), shuffled_target.begin() + test_num),
                     train_target(shuffled_target.begin() + test_num, shuffled_target.end());

    DataFrame x_train(train_data, train_target, head_names, num_features);
    DataFrame x_test(test_data, test_target, head_names, num_features);

    return {x_train, x_test};

}



