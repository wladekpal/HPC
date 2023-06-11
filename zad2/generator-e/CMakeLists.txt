cmake_minimum_required(VERSION 3.5)
project (MIMUW-matgen)

SET(CMAKE_CXX_COMPILER mpicxx)
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "-std=c++14 -g3 -Wall -O0")

LINK_LIBRARIES("blas")


add_executable(ca3dmm
        ca3dmm.cpp
        densematgen.cpp
        densematgen.h
        multiplication.h
        multiplication.cpp)
