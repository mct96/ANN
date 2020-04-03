cmake_minimum_required(VERSION 3.0)

project(ANN VERSION 1.0 LANGUAGES CXX)

add_executable(main main.cpp src/activation_function.cpp src/mlp.cpp)