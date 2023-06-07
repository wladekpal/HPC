#include <cstdio>
#include <iostream>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>

#include "clicque.cu"


int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cout << "Usage ./kcliques graph_input k output" << std::endl;
        return 1;
    }

    std::string input_file_name = std::string(argv[1]);
    int k = std::stoi(argv[2]);
    std::string output_file_name = std::string(argv[3]);

    std::ifstream input_file(input_file_name);

    std::vector<std::pair<int,int>> v;

    int x, y, vertex_count;
    while(input_file >> x >> y) {
        // std::cout << "Siema, " << x << y << std::endl;
        vertex_count = std::max(vertex_count, x);
        vertex_count = std::max(vertex_count, y);
        v.push_back(std::make_pair(x, y));
    }

    std::cout << "vertex_count " << vertex_count << std::endl;

    thrust::host_vector<thrust::host_vector<int>> edges(vertex_count);

    for(int i = 0; i < v.size(); i++) {
        edges[v[i].first].push_back(v[i].second);
    }

    thrust::host_vector<int> result = find_cliques(edges, k);

    std::ofstream output_file(output_file_name);

    for(int i = 0; i < result.size(); i++){
        output_file << result[i] << " ";
    }

    return 0;
}