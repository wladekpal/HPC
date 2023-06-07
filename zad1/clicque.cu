#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/set_operations.h>

static void handleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))

#define TILE_DIM 32
#define BLOCK_ROWS 8

#define MAX_STACK_SIZE 12
#define MAX_DEGREE 1024
#define GROUP_SIZE 32
#define BIN_ENC_SIZE 32
#define FULL_32_FLAG 0xffffffff


// Kernel for induced sub_graph extraction
__global__ void induced_subgraph_extraction_kernel(
        int *edges, // SORTED!!!!!
        int *edge_sizes,
        unsigned int *result,
        int num_v) {
    
    int start_vertex = blockIdx.x; // TODO probably should be in for loop, there can be more verticies in graph than blocks
    // int group_id = threadIdx.x;
    int group_size = blockDim.x;
    int thread_id = threadIdx.x;


    //Here extract induced subgraph 
    __shared__ unsigned int start_edges[MAX_DEGREE];

    for (int i = thread_id; i < edge_sizes[start_vertex]; i++) {
        start_edges[i] = edges[start_vertex * MAX_DEGREE + i]; // TODO: czy tutaj MAX_DEGREE ???
    }

    for (int i = thread_id; i < edge_sizes[start_vertex]; i+= group_size) {
        unsigned int enc = 0;
        unsigned int factor = 0;
        unsigned int counter_neighbour = 0;
        unsigned int counter_start = 0;
        unsigned int current_vertex = edges[start_vertex * MAX_DEGREE + i];

        while(counter_start < edge_sizes[start_vertex] && counter_neighbour < edge_sizes[current_vertex]) {
            if (start_edges[counter_start] == edges[current_vertex * MAX_DEGREE + counter_neighbour]) {
                enc |= 1;
                counter_start++;
                counter_neighbour++;
                enc = enc << 1;
            }
            else if (start_edges[counter_start] < edges[current_vertex * MAX_DEGREE + counter_neighbour]) {
                counter_start++;
                enc = enc << 1;
            }
            else {
                counter_neighbour++;
            }

            if (counter_start % BIN_ENC_SIZE == 31) {
                result[start_vertex * num_v * MAX_DEGREE + i * MAX_DEGREE + factor] = enc;
                enc = 0;
                factor++;
            }
        }

        if (counter_start % BIN_ENC_SIZE != 31) {
                result[start_vertex * num_v * MAX_DEGREE + i * MAX_DEGREE + factor] = enc << (31 - counter_start); //TODO: tutaj chyba jakoś inaczej
                enc = 0;
                factor++;
        }
    }
}

__device__ int get_next_vertex_idx(
        unsigned int *list, // int_32[(MAX_DEGREE / BIN_ENC_SIZE)]
        int *idx, // index of group in list above (from right side)
        unsigned int *mask_idx // Index inside of group (from right side)
    ) {

    while (*idx < (MAX_DEGREE / BIN_ENC_SIZE)) {
        unsigned int x = list[*idx] & (FULL_32_FLAG << *mask_idx);
        x = __ffs(x);

        if (x != 0){
            *mask_idx = x ;
            x = 32 - x;

            if (*mask_idx >= 31) {
                *mask_idx = 0;
                (*idx)++;
            }

            return x;
        }

        (*idx)++;
        *mask_idx = 0;
    }

    return -1;
}


__device__ unsigned int get_number_of_bits(
        unsigned int *list ) {
    unsigned int result = 0;
    for (int i = 0; i < (MAX_DEGREE / BIN_ENC_SIZE); i++) {
        result += __popc(list[i]);
    }

    return result;
}


// Kernel of vertex-centric graph orientation approach                                     
__global__ void graph_orientation_kernel(
        unsigned int *sub_graphs, // int_32[NUM_V][MAX_DEGREE][(MAX_DEGREE / BIN_ENC_SIZE)] TODO: maybe it's better to extract it in this function ???
        unsigned int *num_cliques, 
        unsigned int k,
        unsigned int num_v) {

    // Stack in SHARED memory (pre-allocated)
    // Counters on stack in SHARED memory
    // Vertex lists on stack in GLOBAL memory
    
    int start_vertex = blockIdx.x; // TODO probably should be in for loop, there can be more verticies in graph than blocks
    int group_id = blockIdx.y;
    int group_size = blockDim.x;
    int thread_id = threadIdx.x;

    int verticies[MAX_STACK_SIZE]; // Currently worked vertex index in adjacency list from level above for each recursion depth
    unsigned int masks[MAX_STACK_SIZE];
    unsigned int current_adj_size[MAX_STACK_SIZE];
    unsigned int current_level; // Level of recursion


    __shared__ unsigned int sub_graph_adj[MAX_STACK_SIZE][(MAX_DEGREE / BIN_ENC_SIZE)];
    // Initialization (first depth of recursion)
    current_level = 1;
    verticies[0] = 0;

    for (int i = thread_id; i < (MAX_DEGREE / BIN_ENC_SIZE); i += group_size) {
        sub_graph_adj[0][i] = sub_graphs[start_vertex * num_v * MAX_DEGREE + group_id * MAX_DEGREE + i];
    }
    verticies[0] = 0;
    masks[0] = 0;

    while(true) {
        // Get info about current task
        // int current_vertex_idx = verticies[current_level]; // Index of current vertex on adjacency list from level above
        // unsigned int prev_size = current_adj_size[current_level - 1]; //Size of adjacency list on higher level
        
        // Update verticies
        if (thread_id == 0){
            printf("KERNEL: group: %d id: %d verticies %d masks %d graph: %u\n", group_id, thread_id, verticies[current_level-1], masks[current_level-1], sub_graph_adj[current_level-1][0]);
        }
        int current_vertex_idx = get_next_vertex_idx(sub_graph_adj[current_level-1], &(verticies[current_level-1]), &(masks[current_level-1]));

        if (thread_id == 0){
            printf("KERNEL: group: %d id: %d vertex_idx %d level: %d\n", group_id, thread_id, current_vertex_idx, current_level);
        }
        // If current vertex is beyond the adjacency list from previous level, we have exhausted a level,
        // and need to backtrack.
        if (current_vertex_idx < 0) {
            if (current_level == 0)
                return;

            current_level--;
            

            continue;
        }

        for (int i = thread_id; i < (MAX_DEGREE / BIN_ENC_SIZE); i+= group_size) {
            sub_graph_adj[current_level][i] = sub_graph_adj[current_level-1][i] & sub_graphs[start_vertex * num_v * MAX_DEGREE + current_vertex_idx * MAX_DEGREE + i];
        }

        if (thread_id == 0) {
            printf("KERNEL: group: %d id: %d adjacency list: \n", group_id, thread_id);
            for (int i = 0; i < 2; i++) {
                printf("%u %u %u \n",sub_graph_adj[current_level-1][i], sub_graphs[start_vertex * num_v * MAX_DEGREE + current_vertex_idx * MAX_DEGREE + i], sub_graph_adj[current_level][i]);
            }

            printf("\n");
        }

        // Vertex that the group is working on in current iteration.
        // int current_vertex = group_adj_lists[current_level - 1][current_vertex_idx]; // Vertex id in GRAPH



        unsigned int current_size = get_number_of_bits(sub_graph_adj[current_level]);

        if (thread_id == 0) {
            printf("KERNEL: group: %d id: %d current_size: %u \n", group_id, thread_id, current_size);
        }

       
        if (current_level + 3 == k) { // If we are at k-th level of recursion we have to accumulate an answer and backtrack

            if (thread_id == 0) {
                printf("Clicque found, %u\n", current_size);
                //atomicAdd(num_cliques, current_size);
                
            }

            // Go one level above in recursion
            // current_level--;

            
            // If we have reached the top of recursion it means that we have searched through the whole subgraph

        }
        else if (current_size > 0) { // If there is still something in adjacency list we go deeper

            // Go one level deeper in recursion
            current_level++;
            // Mark that we are starting from the beginnig on this level of recursion
            verticies[current_level] = 0;
            masks[current_level] = 0;
        }
        
    }
}


// Vertex-centric graph orientation approach
thrust::host_vector<int> graph_orientation(thrust::host_vector<thrust::host_vector<int>> edges, int k) {
    std::cout << "GRAPH ORIENTATION BEGININNG" << std::endl;
    int *dev_edges;
    int *dev_sizes;
    unsigned int *dev_num_cliques;
    unsigned int *dev_results;

    int host_edges[edges.size() * MAX_DEGREE];
    int host_sizes[edges.size()];
    unsigned int host_results[edges.size() * MAX_DEGREE * (MAX_DEGREE / BIN_ENC_SIZE)];
    
    cudaMalloc((void**)&dev_edges, sizeof(int) * edges.size() * MAX_DEGREE);
    cudaMalloc((void**)&dev_sizes, sizeof(int) * edges.size());
    cudaMalloc((void**)&dev_results, sizeof(unsigned int) * edges.size() * MAX_DEGREE * (MAX_DEGREE / BIN_ENC_SIZE));
    cudaMalloc((void**)&dev_num_cliques, sizeof(int));

    for (int i = 0; i < edges.size();i++) {
        std::cout << i << ": ";
        for (int j = 0; j < edges[i].size(); j++) {
            std::cout << edges[i][j] << " ";
            host_edges[i * MAX_DEGREE + j] = edges[i][j];
            
        }
        std::cout << " -> " << edges[i].size() << std::endl;
        host_sizes[i] = edges[i].size();
    }

    std::cout << "data copied to host" << std::endl;

    unsigned int num_cliques = 0;

    cudaMemcpy(dev_edges, host_edges, sizeof(int) * edges.size() * MAX_DEGREE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_sizes, host_sizes, sizeof(int) * edges.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_num_cliques, &num_cliques, sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Everything copied to device, starting kernel" << std::endl;

    
    induced_subgraph_extraction_kernel<<<edges.size(),GROUP_SIZE>>>(dev_edges, dev_sizes, dev_results, edges.size());
    cudaCheck(cudaPeekAtLastError());

    cudaDeviceSynchronize();

    cudaMemcpy(host_results, dev_results, edges.size() * MAX_DEGREE * (MAX_DEGREE / BIN_ENC_SIZE), cudaMemcpyDeviceToHost);

    std::cout << "Results for " <<std::endl;
    for (int l = 0; l < edges.size();l++){
        std::cout << "vertex " << l << std::endl;
        for (int i = 0; i < edges.size(); i++) {
            for (int j = 0; j < 2; j++) {
                std::cout << host_results[l * edges.size() * MAX_DEGREE + i * MAX_DEGREE + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    printf("\n\n\n\n\n");

    
    // dim3 dimGrid(edges.size(), MAX_DEGREE, 1);
    dim3 dimGrid(1, 1, 1);
    graph_orientation_kernel<<<dimGrid, GROUP_SIZE>>>(dev_results, &num_cliques, k, edges.size());
    cudaCheck(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    cudaMemcpy(&num_cliques, dev_num_cliques, sizeof(int), cudaMemcpyDeviceToHost);
    printf("NUMBER OF CLIQUES %d\n", num_cliques);

    return thrust::host_vector<int>(10, 1);
}


__global__ void test_kernel() {
    int idx = 0;
    unsigned int mask_idx = 0;
    unsigned int list[(MAX_DEGREE / BIN_ENC_SIZE)];

    list[0] = 0b01000000000000000000000000000010;
    list[1] = 0b00000000000000000000010000000000;
    int result;
    result = get_next_vertex_idx(list, &idx, &mask_idx);
    printf("Result %d\n", result);

    result = get_next_vertex_idx(list, &idx, &mask_idx);
    printf("Result %d\n", result);

    result = get_next_vertex_idx(list, &idx, &mask_idx);
    printf("Result %d\n", result);
}

thrust::host_vector<int> find_cliques(thrust::host_vector<thrust::host_vector<int>> edges, int k) {

    // test_kernel<<<1,1>>>();
    // cudaDeviceSynchronize();
    thrust::host_vector<int> result = graph_orientation(edges, k);


    return thrust::host_vector<int>(10, 1);
}