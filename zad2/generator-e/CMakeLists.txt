cmake_minimum_required(VERSION 3.5)
project (MIMUW-matgen)

SET(CMAKE_CXX_COMPILER mpicxx)
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "-std=c++14 -g3 -Wall -O0 -lblas")


add_executable(generator
        densematgen.cpp
        densematgen.h
        generator_main.cpp)

add_executable(generator_all
        densematgen.cpp
        densematgen.h
        generator_all.cpp)

add_executable(ca3dmm
        ca3dmm.cpp
        densematgen.cpp
        densematgen.h
        multiplication.h
        multiplication.cpp)

# target_compile_options(ca3dmm ${BLAS_LINKER_FLAGS})
# target_link_library(ca3dmm ${BLAS_LIBRARIES})