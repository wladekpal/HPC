#include <iostream>
#include <cstring>
#include <sys/time.h>
#include <vector>
#include <string>
#include <sstream>
#include "multiplication.h"
#include <mpi.h>

std::tuple<int, int, int> solve_ps(int n, int m, int k, double l) { // TOOD: to nieoptymalne
    int total;
    MPI_Comm_size(MPI_COMM_WORLD, &total);

    double total_d = total;

    int min_score = 1000000, max_score = 0;
    std::tuple<int,int,int> solution;

    for (int p_k = 1; p_k <= total; p_k++) {
        for (int p_n = 1; p_n <= total; p_n++) {
            for (int p_m = 1; p_m <= total; p_m++) {
                double num = p_n * p_m * p_k;
                int score = p_m * k * n + p_n * m * k + p_k * m * n;
                if ((p_n % p_m == 0 || p_m % p_n == 0 ) && num >= l * total_d && num <= total_d) {
                    if (score < min_score) {
                        min_score = score;
                        max_score = num;
                        solution = {p_n, p_m, p_k};
                    }
                    else if (score == min_score && max_score < num) {
                        max_score = num;
                        solution = {p_n, p_m, p_k};
                    }
                }
            }
        }
    }

    return solution;
}

void process_arguments(int argc, char *argv[]) {
    int n, m, k;

    n = std::stoi(argv[1]);
    m = std::stoi(argv[2]);
    k = std::stoi(argv[3]);

    std::vector<std::pair<int, int>> seeds;

    for (int i = 4; i < argc; i++) {
        if (std::string(argv[i]) == "-s") {
            auto seeds_string = std::string(argv[i+1]);


            std::string first_str, second_str;
            bool is_first = true;

            for (int j = 0; j < seeds_string.size(); j++) {
                if (seeds_string[j] == ',') {
                    if (!is_first) {
                        seeds.push_back(std::make_pair(std::stoi(first_str), std::stoi(second_str)));
                        first_str.clear();
                        second_str.clear();
                    }

                    is_first = !is_first;
                }
                else {
                    if (is_first) {
                        first_str += seeds_string[j];
                    }
                    else {
                        second_str += seeds_string[j];
                    }
                }
            }

            if (!is_first) {
                seeds.push_back(std::make_pair(std::stoi(first_str), std::stoi(second_str)));
            }
        }
    }


    auto solution = solve_ps(n, m, k, 0.8);

    for (int i =0; i < seeds.size(); i++) {
        multiply(n, m, k, seeds[i].first, seeds[i].second, solution);
    }
}





int main(int argc, char *argv[]) {
    int n, m, k, p_n, p_m, p_k;

    n = std::stoi(argv[1]);
    m = std::stoi(argv[2]);
    k = std::stoi(argv[3]);



    MPI_Init(&argc,&argv);
    process_arguments(argc, argv);
    MPI_Finalize();

    return 0;
}
