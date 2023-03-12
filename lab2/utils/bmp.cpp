/* 
  Special thanks to my Mom, Dad and ChatGPT for making this code possible. 
  This is an utility library that allows to store image both in the form of bitmap and
  compressed version using DFT (Discrete Fourier Transform) coefficients.
  Please treat this code as read-only external dependency - it's knowledge is not even 
  needed to pass the task.
*/

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <chrono>

#define COLORS_COUNT 3 // red, green, blue

typedef void (CompressFun)(const std::vector<uint8_t>&, std::vector<float>&, std::vector<float>&);
typedef void (DecompressFun)(std::vector<uint8_t>&, const std::vector<float>&, const std::vector<float>&);

std::chrono::high_resolution_clock::time_point timeNow() {
  return std::chrono::high_resolution_clock::now();
}

typedef std::chrono::milliseconds millis;
typedef std::chrono::microseconds micro;

#pragma pack(push, 1)
struct BMPHeader {
    /* Just the content of BMP extension header, nothing fancy */
    char header_field[2];
    uint32_t file_size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;
    uint32_t header_size;
    uint32_t width;
    uint32_t height;
    uint16_t planes;
    uint16_t bits_per_pixel;
    uint32_t compression;
    uint32_t image_size;
    uint32_t r_pixels_per_meter;
    uint32_t y_pixels_per_meter;
    uint32_t colors_used;
    uint32_t important_colors;
};
#pragma pack(pop)

struct BMP {
  // RGB[colorIndex] keeps bitmap of a single color. 
  // RBG[colorIndex][column][row] returns value of the specific color at (row, column) position.
  std::vector<std::vector<uint8_t>> RGB[COLORS_COUNT];
  // Xreal[colorIndex][row] keeps list of coefficients for the specific color and row. 
  // List length depends on <accuracy> value.
  std::vector<std::vector<float>> Xreal[COLORS_COUNT], Ximag[COLORS_COUNT];

  BMPHeader header;

  void read(std::string filename) {
    std::ifstream bmpFile(filename, std::ios::binary);

    bmpFile.read(reinterpret_cast<char*>(&header), sizeof(header));
    
    if (header.header_field[0] != 'B' || header.header_field[1] != 'M' || 
        header.bits_per_pixel != 24 || header.compression != 0) {
      std::cerr << "Invalid file format. Must be 24-bit uncompressed BMP" << std::endl;
      exit(1);
    }

    size_t width = header.width, height = header.height;
    
    for (int colorIndex=0; colorIndex<COLORS_COUNT; colorIndex++) {
      RGB[colorIndex].clear();
      for (int c=0; c<height; c++)
        RGB[colorIndex].push_back(std::vector<uint8_t>(width, 0));
    }
    
    int rowPadding = (4 - ((width * 3) % 4)) % 4;

    for (int r = height - 1; r >= 0; r--) { // BMP files are stored upside-down
        for (int c = 0; c < width; c++) {
            for (int colorIndex=0; colorIndex<COLORS_COUNT; colorIndex++) {
              uint8_t value;
              bmpFile.read(reinterpret_cast<char*>(&value), sizeof(value));       
              RGB[colorIndex][r][c] = value;
            }
        }

        bmpFile.seekg(rowPadding, std::ios::cur); // Skip row padding
    }
  }

  void write(std::string filename) {
    std::ofstream bmpFile(filename, std::ios::binary);

    bmpFile.write(reinterpret_cast<char*>(&header), sizeof(header));
    size_t width = header.width, height = header.height;
    
    int rowPadding = (4 - ((width * header.bits_per_pixel / 8) % 4)) % 4; // Pad each row to a multiple of 4 bytes
    
    for (int r = height - 1; r >= 0; r--) { // BMP files are stored upside-down
        for (int c = width - 1; c >= 0; c--) {
            for (int colorIndex=0; colorIndex<COLORS_COUNT; colorIndex++) {
              uint8_t value = RGB[colorIndex][r][c];
              bmpFile.write(reinterpret_cast<char*>(&value), sizeof(value));    
            }
	      }
        uint8_t padding_value = 0;
        bmpFile.write(reinterpret_cast<char*>(&padding_value), sizeof(uint8_t) * rowPadding);
    }
    bmpFile.close();
  }

  float compress(CompressFun *fun, size_t accuracy) {
      float sizeRatio = accuracy * 2.0 / header.width;
      float typeRatio = sizeof(float) / (float) sizeof(uint8_t);
      fprintf(stderr, "Compression ratio %.2f\n", sizeRatio * typeRatio);
      micro totalTime{0};

      for (int colorIndex=0; colorIndex<COLORS_COUNT; colorIndex++) {
          Xreal[colorIndex].clear();
          Ximag[colorIndex].clear();

          for (int j=0; j<header.height; j++) {
            Xreal[colorIndex].push_back(std::vector<float>(accuracy, 0));
            Ximag[colorIndex].push_back(std::vector<float>(accuracy, 0));

            auto startTime = timeNow();
            fun(RGB[colorIndex][j], Xreal[colorIndex][j], Ximag[colorIndex][j]);
            totalTime += std::chrono::duration_cast<micro>(timeNow() - startTime);

            RGB[colorIndex][j].clear();
          }
      }

      return std::chrono::duration_cast<millis>(totalTime).count() / 1000.0;
  }

  float decompress(DecompressFun *fun) {
      micro totalTime{0};

      for (int colorIndex=0; colorIndex<COLORS_COUNT; colorIndex++) {
          for (int j=0; j<header.height; j++) {
            RGB[colorIndex][j].resize(header.width);

            auto startTime = timeNow();
            fun(RGB[colorIndex][j], Xreal[colorIndex][j], Ximag[colorIndex][j]);
          totalTime += std::chrono::duration_cast<micro>(timeNow() - startTime);
          }
      }

      return std::chrono::duration_cast<millis>(totalTime).count() / 1000.0;
  }
};
