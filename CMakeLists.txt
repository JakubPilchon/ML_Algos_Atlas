cmake_minimum_required(VERSION 3.30)
project(Atlas_projekt)

set(CMAKE_CXX_STANDARD 20)

add_executable(Atlas_projekt main.cpp
        dataframe.h
        dataframe.cpp)
