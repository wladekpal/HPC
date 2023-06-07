#include <iostream>
#include <cstring>
#include <sys/time.h>
#include <vector>
#include <string>
#include <sstream>
#include <mpi.h>

void multiply(int n, int m, int k, int seed_A, int seed_B, std::tuple<int, int, int> p_counts);