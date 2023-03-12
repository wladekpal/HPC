#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
#include <complex>
#include <omp.h>
#include <fstream>

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
  // #pragma omp parallel for collapse(2)
  // for (int i = 0; i < valuesCount; i++) { // Tej nie paralelizować
  //     for (int k = 0; k < accuracy; k++) { //tutaj zrobić z nowait, że master liczy swoje wartości a wątki liczą kolejne cos i sin nie czekając na mastera
  //         float theta = (2 * M_PI * k * i) / valuesCount;
  //         coss[i][k] = cos(theta);
  //         sins[i][k] = sin(theta);
  //     }
  // }

  // for (int i = 0; i < valuesCount; i++) {
  //   float total = 0;
  //   for (int k = 0; k < accuracy; k++) {
  //     total += (Xreal[k] * coss[i][k] + Ximag[k] * sins[i][k]);
  //   }

  //   values[i] = total / valuesCount;
  // }

  int threadId;
  float total;

  #pragma omp parallel for
  for (int k = 0; k < accuracy; k++) {
    // float theta = (2 * M_PI * k * i) / valuesCount;
    coss[0][k] = 1;
    sins[0][k] = 0;
  }

  // for (int i = 0; i < valuesCount; i++) { // Tej nie paralelizować
  //   total = 0;
  //   #pragma omp parallel for private(threadId)
  //   for (int k = 0; k < accuracy; k++) { //tutaj zrobić z nowait, że master liczy swoje wartości a wątki liczą kolejne cos i sin nie czekając na mastera
  //     threadId = omp_get_thread_num();
  //     if (threadId == 0) {
  //       total += (Xreal[k] * coss[i][k] + Ximag[k] * sins[i][k]);
  //       // std::cout << total << " " << Xreal[k] << " " << coss[i][k] << " " << Ximag[k] << " " << sins[i][k] << std::endl;
  //     }
  //     else if (k != accuracy - 1) {
  //       float theta = (2 * M_PI * k * (i+1)) / valuesCount;
  //       coss[i+1][k] = cos(theta);
  //       sins[i+1][k] = sin(theta);
  //     }
  //   }
    
  //   values[i] = total / valuesCount;
  // }
  #pragma omp sections
  {
    #pragma omp section
    {
      for (int i = 0; i < valuesCount; i++) {
        float total = 0;
        for (int k = 0; k < accuracy; k++) {
          float theta = (2 * M_PI * k * i) / valuesCount;
          total += Xreal[k] * coss[i][k] + Ximag[k] * sins[i][k];
        }
        values[i] = total / valuesCount;
      }
    }

    #pragma omp section
    {
      #pragma omp parallel for collapse (2)
      for (int i = 1; i < valuesCount; i++) {
        for (int k = 0; k < accuracy; k++) {
          float theta = (2 * M_PI * k * i) / valuesCount;
          sins[i][k] = sin(theta);
          coss[i][k] = cos(theta);
        }
      }
    }
  }


}

int main(int argc, char* argv[]) {
  BMP bmp;
  bmp.read("example.bmp");

  size_t accuracy = 16; // We are interested in values from range [8; 32]
  int threads = 0;

  if (argc > 3) {
    std::cout << "Usage ./dft accuracy number_of_threads" << std::endl;
  }
  
  if (argc >= 2) {
    accuracy = atoi(argv[1]);
  }

  if (argc == 3) {
    threads = atoi(argv[2]);
  }

  std::ofstream result_file;
  result_file.open ("results.csv", std::ios_base::app);
    
  // bmp.{compress,decompress} will run provided function on every bitmap row and color.
  float seqCompressTime = bmp.compress(compress, accuracy);
  float seqDecompressTime = bmp.decompress(decompress);

  printf("Compress time: %.2lfs\nDecompress time: %.2lfs\nTotal: %.2lfs\n", 
    seqCompressTime, seqDecompressTime, seqCompressTime + seqDecompressTime);

  bmp.write("example_result.bmp");

  BMP naive_par_bmp;
  naive_par_bmp.read("example.bmp");

  float naiveCompressTime = naive_par_bmp.compress(compressParNaive, accuracy);
  float decompressTime = naive_par_bmp.decompress(decompresPar);

  printf("Naive Parallel => Compress time: %.2lfs\nDecompress time: %.2lfs\nTotal: %.2lfs\n", 
    naiveCompressTime, decompressTime, naiveCompressTime + decompressTime);

  naive_par_bmp.write("example_result_naive_par.bmp");


  BMP par_bmp;
  par_bmp.read("example.bmp");

  float compressTime = par_bmp.compress(compressPar, accuracy);
  decompressTime = par_bmp.decompress(decompresPar);

  printf("Parallel => Compress time: %.2lfs\nDecompress time: %.2lfs\nTotal: %.2lfs\n", 
    compressTime, decompressTime, compressTime + decompressTime);

  par_bmp.write("example_result_par.bmp");

  result_file << accuracy << "," << threads << ",";
  result_file << seqCompressTime / naiveCompressTime << "," << seqCompressTime / compressTime << "," << seqDecompressTime / decompressTime << std::endl;

  return 0;
}

