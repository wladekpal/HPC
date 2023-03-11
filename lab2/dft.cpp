#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
#include <complex>
#include <omp.h>

#include "utils/bmp.cpp"


void compress(const std::vector<uint8_t> &values, std::vector<float> &Xreal, std::vector<float> &Ximag) {
  // values, Xreal and Ximag are values describing single color of single row of bitmap. 
  // This function will be called once per each (color, row) combination.
  size_t valuesCount = values.size();
  int accuracy = Xreal.size();
  for (int k = 0; k < accuracy; k++) {
      for (int i = 0; i < valuesCount; i++) {
          float theta = (2 * M_PI * k * i) / valuesCount;
          Xreal[k] += values[i] * cos(theta);
          Ximag[k] -= values[i] * sin(theta);
      }
  }
}

void decompress(std::vector<uint8_t> &values, const std::vector<float> &Xreal, const std::vector<float> &Ximag) {
  // values, Xreal and Ximag are values describing single color of single row of bitmap.
  // This function will be called once per each (color, row) combination.
  int accuracy = Xreal.size();
  size_t valuesCount = values.size();

  std::vector<float> rawValues(valuesCount, 0);

  for (int i = 0; i < valuesCount; i++) {
      for (int k = 0; k < accuracy; k++) {
          float theta = (2 * M_PI * k * i) / valuesCount;
          rawValues[i] += Xreal[k] * cos(theta) + Ximag[k] * sin(theta);
      }
      values[i] = rawValues[i] / valuesCount;
  }
}

void compressPar(const std::vector<uint8_t> &values, std::vector<float> &Xreal, std::vector<float> &Ximag) {
  // PUT YOUR IMPLEMENTATION HERE
  size_t valuesCount = values.size();
  int accuracy = Xreal.size();

  float* r_ptr = &Xreal[0];
  float* i_ptr = &Ximag[0];

  #pragma omp parallel for reduction(+:r_ptr[:accuracy]) reduction(-:i_ptr[:accuracy]) collapse(2)
  for (int k = 0; k < accuracy; k++) {
      for (int i = 0; i < valuesCount; i++) {
          float theta = (2 * M_PI * k * i) / valuesCount;
          r_ptr[k] += values[i] * cos(theta);
          i_ptr[k] -= values[i] * sin(theta);
      }
  }
}

void compressParNaive(const std::vector<uint8_t> &values, std::vector<float> &Xreal, std::vector<float> &Ximag) {
  // PUT YOUR IMPLEMENTATION HERE
  size_t valuesCount = values.size();
  int accuracy = Xreal.size();

  #pragma omp parallel for collapse(2)
  for (int k = 0; k < accuracy; k++) {
      for (int i = 0; i < valuesCount; i++) {
          float theta = (2 * M_PI * k * i) / valuesCount;
          #pragma omp atomic update
          Xreal[k] += values[i] * cos(theta);
          #pragma omp atomic update
          Ximag[k] -= values[i] * sin(theta);
      }
  }
}



void decompresPar(std::vector<uint8_t> &values, const std::vector<float> &Xreal, const std::vector<float> &Ximag) {
  // PUT YOUR IMPLEMENTATION HERE
  int accuracy = Xreal.size();
  size_t valuesCount = values.size();

  float coss[valuesCount][accuracy];
  float sins[valuesCount][accuracy];

  //tutaj liczymy cos i sin dla i = 0
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < valuesCount; i++) { // Tej nie paralelizować
      for (int k = 0; k < accuracy; k++) { //tutaj zrobić z nowait, że master liczy swoje wartości a wątki liczą kolejne cos i sin nie czekając na mastera
          float theta = (2 * M_PI * k * i) / valuesCount;
          coss[i][k] = cos(theta);
          sins[i][k] = sin(theta);
      }
  }

  for (int i = 0; i < valuesCount; i++) {
    float total = 0;
    for (int k = 0; k < accuracy; k++) {
      total += (Xreal[k] * coss[i][k] + Ximag[k] * sins[i][k]);
    }

    values[i] = total / valuesCount;
  }

}

int main() {
  BMP bmp;
  bmp.read("example.bmp");

  size_t accuracy = 16; // We are interested in values from range [8; 32]
    
  // bmp.{compress,decompress} will run provided function on every bitmap row and color.
  float compressTime = bmp.compress(compress, accuracy);
  float decompressTime = bmp.decompress(decompress);

  printf("Compress time: %.2lfs\nDecompress time: %.2lfs\nTotal: %.2lfs\n", 
    compressTime, decompressTime, compressTime + decompressTime);

  bmp.write("example_result.bmp");

  BMP naive_par_bmp;
  naive_par_bmp.read("example.bmp");

  compressTime = naive_par_bmp.compress(compressParNaive, accuracy);
  decompressTime = naive_par_bmp.decompress(decompresPar);

  printf("Naive Parallel => Compress time: %.2lfs\nDecompress time: %.2lfs\nTotal: %.2lfs\n", 
    compressTime, decompressTime, compressTime + decompressTime);

  naive_par_bmp.write("example_result_naive_par.bmp");


  BMP par_bmp;
  par_bmp.read("example.bmp");

  compressTime = par_bmp.compress(compressPar, accuracy);
  decompressTime = par_bmp.decompress(decompresPar);

  printf("Parallel => Compress time: %.2lfs\nDecompress time: %.2lfs\nTotal: %.2lfs\n", 
    compressTime, decompressTime, compressTime + decompressTime);

  par_bmp.write("example_result_par.bmp");


  return 0;
}

