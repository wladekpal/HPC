#include <iostream>
#include <cstring>
#include <sys/time.h>
#include <vector>
#include <string>
#include <sstream>
#include <mpi.h>
#include <cassert>
#include "densematgen.h"


bool is_in_range(int x, int y, std::pair<int, int> x_range, std::pair<int,int> y_range){
    return x >= x_range.first && x <= x_range.second && y >= y_range.first && y <= y_range.second;
}


// Chunk of matrix, that is padded, so that it is square
class MatrixChunk {
    public:
        int size;
        double* data;

        MatrixChunk (int size) : size(size) {
            this->data = (double*) calloc(size * size, sizeof(double*));
            
        }

        MatrixChunk (std::pair<int,int> x_gen_range, std::pair<int, int> y_gen_range, std::pair<int,int> x_range, std::pair<int,int> y_range, int seed, int size) : 
                    size(size) {
            
            this->data = (double*) calloc(size * size, sizeof(double*));

            int x_offset = x_range.first;
            int y_offset = y_range.first;

            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    if (is_in_range(i + x_offset, j + y_offset, x_gen_range, y_gen_range)) {
                        this->data[j * size + i] = generate_double(seed, j + y_offset, i + x_offset);
                    }
                }
            }
                
        }

        ~MatrixChunk() {
            free((void*)this->data);
        }

        void print_matrix() {
            std::cout << this->size << "\n";
            for (int i = 0 ; i < size; i++) {
                for (int j = 0 ; j < size; j++) {
                    std::cout << this->data[i * size + j] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
};


MPI_Request send_async_matrix(MPI_Comm comm, int dest, MatrixChunk &M) {
    MPI_Request result;
    MPI_Isend(M.data, M.size * M.size, MPI_DOUBLE, dest, 0, comm, &result);

    return result;
}

MPI_Request recv_async_matrix(MPI_Comm comm, int src, MatrixChunk &M) {
    MPI_Request result;
    MPI_Irecv(M.data, M.size * M.size, MPI_DOUBLE, src, 0, comm, &result);

    return result;
}


//Performs C = C + A * B
void multiply_add(MatrixChunk &A, MatrixChunk &B, MatrixChunk &C) {
    int n = A.size;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for(int k = 0; k < n; k++) {
                C.data[i * n + j] += A.data[i * n + k] * B.data[k * n + j];

                int my_rank;
                MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

                if (my_rank == 0)
                    std::cout << "Proc: " << my_rank << " i:" << i << " j:" << j << " k:" << k << "/n" << A.data[i * n + k] << " " << B.data[k * n + j] << "/n" << std::endl;
            }
        }
    }
}



void cannon(int x, int y, int grid_size, MPI_Comm comm, MatrixChunk &A, MatrixChunk &B, MatrixChunk &C) {
    MPI_Request send_horizontal, send_vertical, recv_horizontal, recv_vertical;
    MPI_Status s;

    int left, right, down, up;

    MPI_Cart_shift(comm, 0, 1, &left, &right);
    MPI_Cart_shift(comm, 1, 1, &down, &up);

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0){
        std::cout << "Before:";
        A.print_matrix();
        B.print_matrix();
        std::cout << "Neighb" << left << " " << right << " " << up << " " << down << std::endl;
    }

    

    for (int i = 0 ; i < grid_size; i++) {

        send_horizontal = send_async_matrix(comm, left , A);
        send_vertical = send_async_matrix(comm, up, B);

        multiply_add(A, B, C);

        MPI_Wait(&send_horizontal, &s);
        MPI_Wait(&send_vertical, &s);

        recv_horizontal = recv_async_matrix(comm, right, A);
        recv_vertical = recv_async_matrix(comm, down, B);

        MPI_Wait(&recv_horizontal, &s);
        MPI_Wait(&recv_vertical, &s);

        if (my_rank == 0){
            std::cout << "After:";
            A.print_matrix();
            B.print_matrix();
        }
    }
}

void multiply(int n, int m, int k, int seed_A, int seed_B, std::tuple<int, int, int> p_counts) {
    int p_n, p_m, p_k, p_total;

    p_n = std::get<0>(p_counts);
    p_m = std::get<1>(p_counts);
    p_k = std::get<2>(p_counts);

    p_total = p_n * p_m * p_k;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        std::cout << "Solution: " << p_n << " " << p_m << " " << p_k << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD); // TODO: wywalić

    if (my_rank >= p_total) {
        return; //TODO: tutaj chyba nie return nie można tak robić bez zrobienia nowego comm worlda!!!!!
    }

    int num_proc_in_k_group = p_n * p_m;

    int my_k_group = my_rank / num_proc_in_k_group;
    int my_id_in_group = my_rank % num_proc_in_k_group;

    int c, x_dim_cannon_group, y_dim_cannon_group, my_id_in_cannon, cannon_size, x_coord_in_group, y_coord_in_group, x_coord_in_cannon, y_coord_in_cannon;
    int B_x_chunk_coord_in_group, B_y_chunk_coord_in_group, B_x_chunk_coord_in_cannon, B_y_chunk_coord_in_cannon;
    int A_x_chunk_coord_in_group, A_y_chunk_coord_in_group, A_x_chunk_coord_in_cannon, A_y_chunk_coord_in_cannon;
    int A_x_chunk_size, A_y_chunk_size, B_x_chunk_size, B_y_chunk_size;
    int x_coord_in_k_group, y_coord_in_k_group, x_group_offset, y_group_offset;
    int my_cannon_group;
    int replication_offset;

    


    std::pair <int,int> A_group_x_range, A_group_y_range, B_group_x_range, B_group_y_range, C_group_x_range, C_group_y_range;

    A_group_x_range = std::make_pair((k / p_k) * my_k_group, (k / p_k + 1) * my_k_group);
    A_group_y_range = std::make_pair(0, m);

    B_group_x_range = std::make_pair(0, n);
    B_group_y_range = std::make_pair((k / p_k) * my_k_group, (k / p_k + 1) * my_k_group);


    std::pair<int,int> A_x_gen_range, A_y_gen_range, B_x_gen_range, B_y_gen_range, C_x_range, C_y_range; // Range of elements that have to be generated by the process
    std::pair<int,int> A_x_real_range, A_y_real_range, B_x_real_range, B_y_real_range; //Range of elements that the process has to have to start multiplication

    if (p_n >= p_m) {
        c = p_n / p_m;
        x_dim_cannon_group = p_n / c;
        y_dim_cannon_group = p_m;

        cannon_size = x_dim_cannon_group * y_dim_cannon_group;
        my_cannon_group = my_id_in_group / cannon_size;
        my_id_in_cannon = my_id_in_group % cannon_size;


        // Calculate A ranges
        x_group_offset = my_k_group * (k / p_k);
        y_group_offset = 0;
        
        A_x_chunk_size = (k / p_k) / (p_n / c);
        A_y_chunk_size = m / p_m;

        
        x_coord_in_group = my_id_in_cannon / y_dim_cannon_group;
        y_coord_in_group = my_id_in_cannon % y_dim_cannon_group;

        if (c > 1)
            x_coord_in_cannon = x_coord_in_group % c;
        else
            x_coord_in_cannon = x_coord_in_group;
        y_coord_in_cannon = y_coord_in_group;


        A_x_gen_range = std::make_pair(my_cannon_group * A_x_chunk_size + x_group_offset, (my_cannon_group + 1) * A_x_chunk_size + x_group_offset);
        A_y_gen_range = std::make_pair(y_coord_in_cannon * A_y_chunk_size + y_group_offset, (y_coord_in_cannon + 1) * A_y_chunk_size + y_group_offset);

        A_x_chunk_size = A_x_chunk_size * c; // This has to be expanded, to store replicated block

        A_x_real_range = A_x_gen_range;
        if (A_x_real_range.first % A_x_chunk_size != 0)
            A_x_real_range.first = A_x_real_range.first - (A_x_real_range.first % A_x_chunk_size);
        if (A_x_real_range.second % A_x_chunk_size != 0)
            A_x_real_range.second = A_x_real_range.second - (A_x_real_range.second % A_x_chunk_size) + A_x_chunk_size;

        A_y_real_range = A_y_gen_range;

        

        // Calculate B ranges
        x_coord_in_group = my_id_in_group / p_m;
        y_coord_in_group = my_id_in_group % p_m;

        x_group_offset = 0;
        y_group_offset = my_k_group * (k / p_k);

        B_x_chunk_size = n / p_n;
        B_y_chunk_size = (k / p_k) / p_m;

        B_x_gen_range = std::make_pair(x_coord_in_group * B_x_chunk_size + x_group_offset, (x_coord_in_group + 1) * B_x_chunk_size + x_group_offset);
        B_y_gen_range = std::make_pair(y_coord_in_group * B_y_chunk_size + y_group_offset, (y_coord_in_group + 1) * B_y_chunk_size + y_group_offset);

        B_x_real_range = B_x_gen_range;
        B_y_real_range = B_y_gen_range;

        std::cout << "Proc: " << my_rank << "| k_group: " << my_k_group << "| my_id_in_k: " << my_id_in_group << "| cannon_group: " << my_cannon_group << "| my_id_in_cannon: " << my_id_in_cannon; 
        std::cout << "| coords in cannon: (" << x_coord_in_cannon << "," << y_coord_in_cannon << ")";
        std::cout << "| coords in group: (" << x_coord_in_group << "," << y_coord_in_group << ")";
        std::cout << "| A range x: (" << A_x_gen_range.first << "," << A_x_gen_range.second << ") y:(" << A_y_gen_range.first << "," << A_y_gen_range.second << ") | B range x:(" << B_x_gen_range.first << "," << B_x_gen_range.second << ") y: (" << B_y_gen_range.first << "," << B_y_gen_range.second << ")";
        std::cout << "| A_x_real_range: (" << A_x_real_range.first << "," << A_x_real_range.second << ")"<< std::endl;
    }
    else {
        c = p_m / p_n;
        x_dim_cannon_group = p_n;
        y_dim_cannon_group = p_m / c;
    }

    
    MPI_Barrier(MPI_COMM_WORLD); // TODO: wywalić

    if (my_rank == 0) {
        std::cout << "-----------------------------------------------------------------------------------" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD); // TODO: wywalić

    MPI_Comm cannon_comm, cannon_comm_grid;
    MPI_Comm_split(MPI_COMM_WORLD, my_k_group * c + my_cannon_group, my_id_in_cannon, &cannon_comm);
    MPI_Barrier(MPI_COMM_WORLD); // TODO: wywalić

    int dims[2] = {x_dim_cannon_group, y_dim_cannon_group};
    int periods[2] = {1, 1};
    MPI_Cart_create(cannon_comm, 2, dims, periods, false, &cannon_comm_grid);

    int coords[2];
    MPI_Barrier(MPI_COMM_WORLD); // TODO: wywalić

    MPI_Cart_coords(cannon_comm_grid, my_id_in_cannon, 2, coords);

    std::cout << "Proc " << my_rank << "| id_in_cannon: " << my_id_in_cannon << "| coords: " << coords[0] << " " << coords[1] << std::endl; 

    MPI_Barrier(MPI_COMM_WORLD); // TODO: wywalić

    

    int max_chunk_size = std::max(std::max(A_x_chunk_size, A_y_chunk_size), std::max(B_x_chunk_size, B_y_chunk_size)); // TODO: czy tutaj nie można jakoś lepiej tego zrobić ?

    A_x_gen_range.second -= 1;
    A_y_gen_range.second -= 1;
    A_x_real_range.second -= 1;
    A_y_real_range.second -= 1;

    B_x_gen_range.second -= 1;
    B_y_gen_range.second -= 1;
    B_x_real_range.second -= 1;
    B_y_real_range.second -= 1;

    MatrixChunk A(A_x_gen_range, A_y_gen_range, A_x_real_range, A_y_real_range, seed_A, max_chunk_size);
    MatrixChunk B(B_x_gen_range, B_y_gen_range, B_x_real_range, B_y_real_range, seed_B, max_chunk_size);

    if (my_rank == 0) {
        A.print_matrix();
    }

    MPI_Comm redistribution_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_k_group * p_n * p_m + x_coord_in_cannon * p_m + y_coord_in_cannon, 0, &redistribution_comm);

    // TODO: A trzeba zamalocować lepiej, tak, żeby to od razu allgatherem się tam zapisało albo mpi_allreduce
    double* matrix_send = A.data;//(double*) malloc(sizeof(double) * max_chunk_size * max_chunk_size);
    double* matrix_recv = (double*) malloc(sizeof(double) * max_chunk_size * max_chunk_size * c);

    MPI_Barrier(MPI_COMM_WORLD);


    if (c > 1){
        MPI_Allgather(matrix_send, max_chunk_size * max_chunk_size, MPI_DOUBLE, matrix_recv, max_chunk_size * max_chunk_size, MPI_DOUBLE, redistribution_comm);

        for (int i = 0; i < max_chunk_size * max_chunk_size; i++) {
            double sum = 0;
            for (int j = 0; j < c; j++) {
                sum += matrix_recv[max_chunk_size * max_chunk_size * j + i];
            }
            A.data[i] = sum;
        }
    }

    

    if (my_rank >= 0) {
        std::cout << "==================================================================================\n";
        std::cout << "My_rank" <<  my_rank << "\n";
        A.print_matrix();
        B.print_matrix();
         std::cout << "==================================================================================\n";

        std::cout << std::endl;
    }


    MatrixChunk C(max_chunk_size);

    //TODO: redistribution

    cannon(coords[0], coords[1], x_dim_cannon_group, cannon_comm_grid, A, B, C);

    if (my_rank == 0) {
        std::cout << "-----------------------------------------------------------------------------------" << std::endl;
        C.print_matrix();
    }

}