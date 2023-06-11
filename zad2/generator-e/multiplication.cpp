#include <iostream>
#include <cstring>
#include <sys/time.h>
#include <vector>
#include <string>
#include <sstream>
#include <cblas.h>
#include <tuple>
#include <mpi.h>
#include <cassert>
#include "densematgen.h"


int divide_ceil(int a, int b) {
    return (a + b - 1) / b;
}


bool is_in_range(int x, int y, std::pair<int, int> x_range, std::pair<int,int> y_range){
    return x >= x_range.first && x <= x_range.second && y >= y_range.first && y <= y_range.second;
}


class MatrixChunk {
    public:
        int x_size, y_size;
        double* data;

        // This version of constructor creates empty chunk
        MatrixChunk (int x_size, int y_size) : x_size(x_size), y_size(y_size) {
            this->data = (double*) calloc(x_size * y_size, sizeof(double*));
        }

        // This version of constructor generates numbers in chunk between specified ranges
        MatrixChunk (std::pair<int,int> x_gen_range, std::pair<int, int> y_gen_range, std::pair<int,int> x_range, std::pair<int,int> y_range, int seed, int x_size, int y_size, bool transpose) : 
                    x_size(x_size),
                    y_size(y_size) {
            
            this->data = (double*) calloc(x_size * y_size, sizeof(double*));

            for (int row = y_gen_range.first; row <= y_gen_range.second; row++) {
                for (int column = x_gen_range.first; column <= x_gen_range.second; column++) {
                    int y = row - y_range.first;
                    int x = column - x_range.first;

                    this->data[y * this->x_size + x] = (transpose) ? generate_double(seed, column, row) : generate_double(seed, row, column);
                }
            }
                
        }

        ~MatrixChunk() {
            free((void*)this->data);
        }
};


void print_row(double* data, int size) {
    for (int i = 0 ; i < size; i++)
        std::cout << data[i] << " ";
}

void print_col(double* data, int x_size, int y_size) {
    for (int i = 0; i < y_size; i++)
        std::cout << data[i * x_size] << " ";
}


MPI_Request send_async_matrix(MPI_Comm comm, int dest, MatrixChunk &M) {
    MPI_Request result;
    MPI_Isend(M.data, M.x_size * M.y_size, MPI_DOUBLE, dest, 0, comm, &result);

    return result;
}

MPI_Request recv_async_matrix(MPI_Comm comm, int src, MatrixChunk &M) {
    MPI_Request result;
    MPI_Irecv(M.data, M.x_size * M.y_size, MPI_DOUBLE, src, 0, comm, &result);

    return result;
}


//Performs C = C + A * B
void multiply_add(MatrixChunk &A, MatrixChunk &B, MatrixChunk &C) {
    // for (int i = 0; i < A.y_size; i++) {
    //     for (int j = 0; j < B.x_size; j++) {
    //         for(int k = 0; k < A.x_size; k++) {
    //             C.data[i * B.x_size + j] += A.data[i * A.x_size + k] * B.data[k * B.x_size + j];
    //         }
    //     }
    // }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.y_size, B.x_size, A.x_size, 1.0, A.data, A.x_size, B.data, B.x_size, 1.0, C.data, B.x_size);
}



void cannon(int grid_size, MPI_Comm comm, MatrixChunk &A, MatrixChunk &B, MatrixChunk &C, int skew_vertical, int skew_horizontal) {
    if (grid_size == 1) {
        multiply_add(A, B, C);
        return;
    }
    
    MPI_Request send_horizontal, send_vertical, recv_horizontal, recv_vertical;
    MPI_Status s;

    int left, right, down, up, skew_left, skew_right, skew_up, skew_down;

    MPI_Cart_shift(comm, 0, 1, &left, &right);
    MPI_Cart_shift(comm, 1, 1, &up, &down);
    MPI_Cart_shift(comm, 0, skew_horizontal, &skew_left, &skew_right);
    MPI_Cart_shift(comm, 1, skew_vertical, &skew_up, &skew_down);


    if (skew_horizontal > 0) {
        MPI_Sendrecv_replace(A.data, A.x_size * A.y_size, MPI_DOUBLE, skew_left, 0, skew_right, 0, comm, &s);
    }

    if (skew_vertical > 0) {
        MPI_Sendrecv_replace(B.data, B.x_size * B.y_size, MPI_DOUBLE, skew_up, 0, skew_down, 0, comm, &s);
    }

    for (int i = 0 ; i < grid_size; i++) {
        // Here we use asynchronous communication
        // At the same thime we send matricies, receive matricies and calculate current product
        send_horizontal = send_async_matrix(comm, left, A);
        send_vertical = send_async_matrix(comm, up, B);

        MatrixChunk new_A(A.x_size, A.y_size);
        MatrixChunk new_B(B.x_size, B.y_size);

        recv_horizontal = recv_async_matrix(comm, right, new_A);
        recv_vertical = recv_async_matrix(comm, down, new_B);

        multiply_add(A, B, C);

        MPI_Wait(&send_horizontal, &s);
        MPI_Wait(&send_vertical, &s);
        MPI_Wait(&recv_horizontal, &s);
        MPI_Wait(&recv_vertical, &s);

        std::swap(A.data, new_A.data);
        std::swap(B.data, new_B.data);
    }
}

void multiply(int n, int m, int k, int seed_A, int seed_B, std::tuple<int, int, int> p_counts, bool print_matrix, double ge_value, bool transpose) {
    int p_n, p_m, p_k, p_total;

    p_n = std::get<0>(p_counts);
    p_m = std::get<1>(p_counts);
    p_k = std::get<2>(p_counts);

    // We assume that p_n >= p_m, so that there is less if statements in the code
    // If original p_m > p_n the dimensions should be transposed before the call and `transpose` flag should be set to true
    assert(p_n >= p_m); 

    p_total = p_n * p_m * p_k;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Comm global_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_rank >= p_total, my_rank, &global_comm);

    if (my_rank >= p_total) {
        return;
    }


    // Constants used to determine location of the process in the grid
    int c = p_n / p_m;
    int num_proc_in_k_group = p_n * p_m;
    int my_k_group = my_rank / num_proc_in_k_group;
    int my_id_in_group = my_rank % num_proc_in_k_group;
    int cannon_size = p_m * p_m;
    int my_cannon_group = my_id_in_group / cannon_size;
    int my_id_in_cannon = my_id_in_group % cannon_size;
    int x_coord_in_group = my_id_in_group / p_m;
    int y_coord_in_group = my_id_in_group % p_m;
    int x_coord_in_cannon = x_coord_in_group % p_m;
    int x_group_offset = my_k_group * divide_ceil(k, p_k);
    int y_group_offset = my_k_group * divide_ceil(k, p_k);


    // Below we calculate sizes and ranges of chunks of matricies required by each process
    int A_x_chunk_size, A_y_chunk_size, B_x_chunk_size, B_y_chunk_size, C_x_chunk_size, C_y_chunk_size; // Sizes of matricies chunks stored by each process
    int A_y_gen_chunk_size; // Size of chunk of A matrix that the process has to generate
    std::pair<int,int> A_x_gen_range, A_y_gen_range, B_x_gen_range, B_y_gen_range; // Range of elements that have to be generated by the process
    std::pair<int,int> A_x_real_range, A_y_real_range, B_x_real_range, B_y_real_range; //Range of elements that the process has to have to start multiplication

    // Calculate A ranges
    A_x_chunk_size = divide_ceil(k, p_k * p_m);
    A_y_chunk_size = divide_ceil(m, p_m);

    A_y_gen_chunk_size = divide_ceil(A_y_chunk_size, c);

    A_x_gen_range = std::make_pair(x_coord_in_cannon * A_x_chunk_size + x_group_offset, (x_coord_in_cannon + 1) * A_x_chunk_size + x_group_offset);
    A_y_gen_range = std::make_pair(y_coord_in_group * A_y_chunk_size + my_cannon_group * A_y_gen_chunk_size, y_coord_in_group * A_y_chunk_size + (my_cannon_group + 1) * A_y_gen_chunk_size);
    A_x_gen_range.second = std::min(A_x_gen_range.second, k);
    A_y_gen_range.second = std::min(A_y_gen_range.second, m);

    A_x_real_range = std::make_pair(x_coord_in_cannon * A_x_chunk_size + x_group_offset, (x_coord_in_cannon + 1) * A_x_chunk_size + x_group_offset);
    A_y_real_range = std::make_pair(y_coord_in_group * A_y_chunk_size, (y_coord_in_group + 1) * A_y_chunk_size);

    A_x_gen_range.second -= 1;
    A_y_gen_range.second -= 1;
    A_x_real_range.second -= 1;
    A_y_real_range.second -= 1;


    // Calculate B ranges
    B_x_chunk_size = divide_ceil(n, p_n);
    B_y_chunk_size = A_x_chunk_size;

    B_x_gen_range = std::make_pair(x_coord_in_group * B_x_chunk_size, (x_coord_in_group + 1) * B_x_chunk_size);
    B_y_gen_range = std::make_pair(y_coord_in_group * B_y_chunk_size + y_group_offset, (y_coord_in_group + 1) * B_y_chunk_size + y_group_offset);

    B_x_real_range = B_x_gen_range;
    B_y_real_range = B_y_gen_range;

    B_x_real_range.second = std::min(B_x_real_range.second, n);
    B_y_real_range.second = std::min(B_y_real_range.second, k);

    B_x_gen_range.second -= 1;
    B_y_gen_range.second -= 1;
    B_x_real_range.second -= 1;
    B_y_real_range.second -= 1;


    // For C we only need chunk sizes
    C_x_chunk_size = divide_ceil(n, p_n);
    C_y_chunk_size = divide_ceil(m, p_m);


    // After all the ranges are calculated we can generate chunks of matricies A and B
    MatrixChunk A(A_x_gen_range, A_y_gen_range, A_x_real_range, A_y_real_range, seed_A, A_x_chunk_size, A_y_chunk_size, transpose);
    MatrixChunk B(B_x_gen_range, B_y_gen_range, B_x_real_range, B_y_real_range, seed_B, B_x_chunk_size, B_y_chunk_size, transpose);
    MatrixChunk C(B_x_chunk_size, A_y_chunk_size);


    // We set up a grid communicator for each cannon group to allow for easy communication between neighbours
    MPI_Comm cannon_comm, cannon_comm_grid;
    MPI_Comm_split(global_comm, my_k_group * c + my_cannon_group, my_id_in_cannon, &cannon_comm);

    int dims[2] = {p_m, p_m};
    int periods[2] = {1, 1};
    MPI_Cart_create(cannon_comm, 2, dims, periods, false, &cannon_comm_grid);

    int coords[2];
    MPI_Cart_coords(cannon_comm_grid, my_id_in_cannon, 2, coords);


    // If there are multiple Cannon groups in each k-group we have to combine pieces of matricies from each cannon
    if (c > 1) {
        MPI_Comm redistribution_comm;
        MPI_Comm_split(global_comm, A_x_real_range.first * m + A_y_real_range.first, 0, &redistribution_comm);

        double* matrix_recv = (double*) malloc(sizeof(double) * A.x_size * A.y_size);
        MPI_Allreduce(A.data, matrix_recv, A.x_size * A.y_size, MPI_DOUBLE, MPI_SUM, redistribution_comm);
        free(A.data);
        A.data = matrix_recv;
    }

    
    // Here every process has all necessary matrix pieces, so we can perform Cannon's algorithm
    cannon(p_m, cannon_comm_grid, A, B, C, coords[0], coords[1]);


    // If there are multiple k-groups we have to sum up the results from them
    if (p_k > 1){
        MPI_Comm reduction_comm;
        MPI_Comm_split(global_comm, y_coord_in_group * (m / p_m) + x_coord_in_group, 0, &reduction_comm);

        double* reduced_C = (double*) malloc(C.x_size * C.y_size * sizeof(double));
        MPI_Allreduce(C.data, reduced_C, C.x_size * C.y_size, MPI_DOUBLE, MPI_SUM, reduction_comm);
        free(C.data);
        C.data = reduced_C;
    }

    
    // Depending on the user input we either print the whole matrix or check how many elements greater than given constant are there
    if (print_matrix) {
        if (my_rank == 0) {
            if (transpose) {
                double* recv_col = (double*) malloc(C_y_chunk_size * sizeof(double));
                MPI_Status s;
                std::cout << n << " " << m << std::endl;
                for (int p_col = 0; p_col < p_n; p_col++) {
                    for (int col = p_col * C_x_chunk_size; col < (p_col + 1) * C_x_chunk_size; col++) {
                        int col_in_proc = col % C_x_chunk_size;
                        int printed_in_row = 0;
                        for (int p_row = 0; p_row < p_m; p_row++) {
                            int proc_num = p_col * p_m + p_row;
                            int to_print;
                            if (proc_num == 0) {
                                print_col(C.data + col_in_proc, C_x_chunk_size, C_y_chunk_size);
                            }
                            else {
                                to_print = std::min(printed_in_row + C_y_chunk_size, m) - printed_in_row;
                                MPI_Recv(recv_col, C_y_chunk_size, MPI_DOUBLE, proc_num, 0, global_comm, &s);
                                print_row(recv_col, to_print);
                            }
                            printed_in_row += C_y_chunk_size;
                        }
                        std::cout << std::endl;
                    }   
                    
                }

                free(recv_col);
            }
            else {
                double* recv_row = (double*) malloc(C_x_chunk_size * sizeof(double));
                MPI_Status s;
                std::cout << m << " " << n << std::endl;
                for (int p_row = 0; p_row < p_m; p_row++) {
                    for (int row = p_row * C_y_chunk_size; row < (p_row + 1) * C_y_chunk_size; row++) {
                        int row_in_proc = row % C_y_chunk_size;
                        int printed_in_row = 0;
                        for (int p_col = 0; p_col < p_n; p_col++) {
                            int proc_num = p_col * p_m + p_row;
                            int to_print = std::min(printed_in_row + C_x_chunk_size, n) - printed_in_row;
                    
                            if (proc_num == 0) {
                                print_row(C.data + row_in_proc * C.x_size, to_print);
                            }
                            else {
                                MPI_Recv(recv_row, C_x_chunk_size, MPI_DOUBLE, proc_num, 0, global_comm, &s);
                                print_row(recv_row, to_print);
                            }
                            printed_in_row += C_x_chunk_size;
                        }
                        std::cout << std::endl;
                    }
                    
                }

                free(recv_row);
            }
        }
        else {
            if (transpose) {
                double* column_buf = (double*) malloc(C_y_chunk_size * sizeof(double));
                for (int column = 0; column < C_x_chunk_size; column++){
                    for (int row = 0; row < C_y_chunk_size; row++) {
                        column_buf[row] = C.data[row * C_x_chunk_size + column];
                    }

                    MPI_Send(column_buf, C_y_chunk_size, MPI_DOUBLE, 0, 0, global_comm);
                }

                free(column_buf);
            }
            else {
                for (int row = 0; row < C_y_chunk_size; row++) {
                    MPI_Send(C.data + row * C_x_chunk_size, C_x_chunk_size, MPI_DOUBLE, 0, 0, global_comm);
                }
            }
        }
    }
    else {
        // Each process participates in counting on it's own submatrix
        int num_ge = 0;
        int ge_chunk_size = divide_ceil(C.y_size, p_k);
        
        for (int row = my_k_group * ge_chunk_size; row < std::min((my_k_group + 1) * ge_chunk_size, C.y_size) ; row++) {
            for (int column = 0; column < C.x_size; column++) {
                if (C.data[row * C.x_size + column] >= ge_value) {
                    num_ge++;
                }
            }
        }

        int result;
        MPI_Reduce(&num_ge, &result, 1, MPI_INT, MPI_SUM, 0, global_comm);

        if (my_rank == 0)
            std::cout << result << std::endl;
    }
}