#include <iostream>
#include <cstring>
#include <sys/time.h>
#include <vector>
#include <string>
#include <sstream>
#include <cblas.h>
#include <mpi.h>
#include <cassert>
#include "densematgen.h"


int divide_ceil(int a, int b) {
    return (a + b - 1) / b;
}


bool is_in_range(int x, int y, std::pair<int, int> x_range, std::pair<int,int> y_range){
    return x >= x_range.first && x <= x_range.second && y >= y_range.first && y <= y_range.second;
}


// Chunk of matrix, that is padded, so that it is square
class MatrixChunk {
    public:
        int x_size, y_size;
        double* data;

        MatrixChunk (int x_size, int y_size) : x_size(x_size), y_size(y_size) {
            this->data = (double*) calloc(x_size * y_size, sizeof(double*));
        }

        MatrixChunk (std::pair<int,int> x_gen_range, std::pair<int, int> y_gen_range, std::pair<int,int> x_range, std::pair<int,int> y_range, int seed, int x_size, int y_size, bool transpose) : 
                    x_size(x_size),
                    y_size(y_size) {
            
            this->data = (double*) calloc(x_size * y_size, sizeof(double*));

            int x_offset = x_range.first;
            int y_offset = y_range.first;

            for (int row = 0; row < y_size; row++) {
                for (int column = 0; column < x_size; column++) {
                    if (is_in_range(column + x_offset, row + y_offset, x_gen_range, y_gen_range) && is_in_range(column + x_offset, row + y_offset, x_range, y_range)) {
                        if (transpose)
                            this->data[row * x_size + column] = generate_double(seed, column + x_offset, row + y_offset);
                        else
                            this->data[row * x_size + column] = generate_double(seed, row + y_offset, column + x_offset);
                    }
                }
            }

            // for (int row = y_range.first; row < y_range.second; row++) {
            //     for (int column = x_range.first; column < x_range.second; column++) {
            //         if (transpose)
            //             this->data[(row - y_offset) * size + column - x_offset] = generate_double(seed, column, row);
            //         else
            //             this->data[(row - y_offset) * size + column - x_offset] = generate_double(seed, row, column);
            //     }
            // }
                
        }

        ~MatrixChunk() {
            free((void*)this->data);
        }

        void print_matrix() {
            std::cout << this->y_size << " " << this->x_size << "\n";
            for (int i = 0 ; i < y_size; i++) {
                for (int j = 0 ; j < x_size; j++) {
                    std::cout << this->data[i * x_size + j] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
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

                    // multiply_add(A, B, C);
                    // MPI_Sendrecv_replace(A.data, A.x_size * A.y_size, MPI_DOUBLE, left, 0, right, 0, comm, &s);
                    // MPI_Sendrecv_replace(B.data, B.x_size * B.y_size, MPI_DOUBLE, up, 0, down, 0, comm, &s);
    }
}

void multiply(int n, int m, int k, int seed_A, int seed_B, std::tuple<int, int, int> p_counts, bool print_matrix, double ge_value, bool transpose) {
    int p_n, p_m, p_k, p_total;

    p_n = std::get<0>(p_counts);
    p_m = std::get<1>(p_counts);
    p_k = std::get<2>(p_counts);

    assert(p_n >= p_m); 


    p_total = p_n * p_m * p_k;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // if (my_rank == 0) {
    //     std::cout << "Solution: " << p_n << " " << p_m << " " << p_k << std::endl;
    // }

    if (my_rank >= p_total) {
        return;
    }
    
    MPI_Comm global_comm;
    MPI_Comm_split(MPI_COMM_WORLD, 0, my_rank, &global_comm);

    int num_proc_in_k_group = p_n * p_m;

    int my_k_group = my_rank / num_proc_in_k_group;
    int my_id_in_group = my_rank % num_proc_in_k_group;

    int c, my_id_in_cannon, cannon_size, x_coord_in_group, y_coord_in_group, x_coord_in_cannon, y_coord_in_cannon;
    int A_x_chunk_size, A_y_chunk_size, B_x_chunk_size, B_y_chunk_size;
    int x_group_offset, y_group_offset;
    int my_cannon_group;


    std::pair<int,int> A_x_gen_range, A_y_gen_range, B_x_gen_range, B_y_gen_range; // Range of elements that have to be generated by the process
    std::pair<int,int> A_x_real_range, A_y_real_range, B_x_real_range, B_y_real_range; //Range of elements that the process has to have to start multiplication
    std::pair<int,int> C_x_range, C_y_range;

    if (p_n >= p_m) {
        c = p_n / p_m;

        cannon_size = p_m * p_m;
        my_cannon_group = my_id_in_group / cannon_size;
        my_id_in_cannon = my_id_in_group % cannon_size;


        // Calculate A ranges
        x_group_offset = my_k_group * (k / p_k);
        y_group_offset = 0;
        
        A_x_chunk_size = (k / p_k) / p_m; // TODO: tutaj pewnie też zaokrąglenie w górę ??
        A_y_chunk_size = m / p_m;

        
        x_coord_in_group = my_id_in_group / p_m;
        y_coord_in_group = my_id_in_group % p_m;


        x_coord_in_cannon = x_coord_in_group % p_m;
        y_coord_in_cannon = y_coord_in_group;

        int A_x_gen_chunk_size = A_x_chunk_size;//divide_ceil(A_x_chunk_size, 1);
        int A_y_gen_chunk_size = divide_ceil(A_y_chunk_size, c);

        // if (my_rank == 0)
        //     std::cout << "A_GEN_CHUNK " << A_x_gen_chunk_size << " " << A_y_gen_chunk_size << std::endl;

        A_x_gen_range = std::make_pair(x_group_offset + x_coord_in_cannon * A_x_gen_chunk_size, x_group_offset + (x_coord_in_cannon + 1) * A_x_gen_chunk_size);
        A_y_gen_range = std::make_pair(y_coord_in_group * A_y_chunk_size + my_cannon_group * A_y_gen_chunk_size + y_group_offset, y_coord_in_group * A_y_chunk_size + (my_cannon_group + 1) * A_y_gen_chunk_size + y_group_offset);
        A_x_gen_range.second = std::min(A_x_gen_range.second, k);
        A_y_gen_range.second = std::min(A_y_gen_range.second, m);

        A_x_real_range = std::make_pair(x_coord_in_cannon * A_x_chunk_size + x_group_offset, (x_coord_in_cannon + 1) * A_x_chunk_size + x_group_offset);
        A_y_real_range = std::make_pair(y_coord_in_group * A_y_chunk_size + y_group_offset, (y_coord_in_group + 1) * A_y_chunk_size + y_group_offset);


        // Calculate B ranges

        x_group_offset = 0;
        y_group_offset = my_k_group * (k / p_k);

        B_x_chunk_size = divide_ceil(n, p_n);
        B_y_chunk_size = A_x_chunk_size;

        B_x_gen_range = std::make_pair(x_coord_in_group * B_x_chunk_size + x_group_offset, (x_coord_in_group + 1) * B_x_chunk_size + x_group_offset);
        B_y_gen_range = std::make_pair(y_coord_in_group * B_y_chunk_size + y_group_offset, (y_coord_in_group + 1) * B_y_chunk_size + y_group_offset);

        B_x_real_range = B_x_gen_range;
        B_y_real_range = B_y_gen_range;

        B_x_real_range.second = std::min(B_x_real_range.second, n);
        B_y_real_range.second = std::min(B_y_real_range.second, k);

        // std::cout << "Proc: " << my_rank << "| k_group: " << my_k_group << "| my_id_in_k: " << my_id_in_group << "| cannon_group: " << my_cannon_group << "| my_id_in_cannon: " << my_id_in_cannon; 
        // std::cout << "| coords in cannon: (" << x_coord_in_cannon << "," << y_coord_in_cannon << ")";
        // std::cout << "| coords in group: (" << x_coord_in_group << "," << y_coord_in_group << ")";
        // std::cout << "| A range x: (" << A_x_gen_range.first << "," << A_x_gen_range.second << ") y:(" << A_y_gen_range.first << "," << A_y_gen_range.second << ") | B range x:(" << B_x_real_range.first << "," << B_x_real_range.second << ") y: (" << B_y_real_range.first << "," << B_y_real_range.second << ")";
        // std::cout << "| A_x_real_range: (" << A_x_real_range.first << "," << A_x_real_range.second << ") ";
        // std::cout << "| A_y_real_range: (" << A_y_real_range.first << "," << A_y_real_range.second << ")" << std::endl;
    }
    else {
        assert(false);
    }


    MPI_Comm cannon_comm, cannon_comm_grid;
    MPI_Comm_split(global_comm, my_k_group * c + my_cannon_group, my_id_in_cannon, &cannon_comm);
    // std::cout << my_rank << "->>>>>>" << my_k_group * c + my_cannon_group << std::endl;

    int dims[2] = {p_m, p_m};
    int periods[2] = {1, 1};
    MPI_Cart_create(cannon_comm, 2, dims, periods, false, &cannon_comm_grid);

    int coords[2];

    MPI_Cart_coords(cannon_comm_grid, my_id_in_cannon, 2, coords);

    

    int max_chunk_size = std::max(std::max(A_x_chunk_size, A_y_chunk_size), std::max(B_x_chunk_size, B_y_chunk_size)); // TODO: czy tutaj nie można jakoś lepiej tego zrobić ?

    A_x_gen_range.second -= 1;
    A_y_gen_range.second -= 1;
    A_x_real_range.second -= 1;
    A_y_real_range.second -= 1;

    B_x_gen_range.second -= 1;
    B_y_gen_range.second -= 1;
    B_x_real_range.second -= 1;
    B_y_real_range.second -= 1;

    MatrixChunk A(A_x_gen_range, A_y_gen_range, A_x_real_range, A_y_real_range, seed_A, A_x_chunk_size, A_y_chunk_size, transpose);
    MatrixChunk B(B_x_gen_range, B_y_gen_range, B_x_real_range, B_y_real_range, seed_B, B_x_chunk_size, B_y_chunk_size, transpose);


    // if (my_rank == 0) {
    //     std::cout << "==================================================================================\n";
    //     std::cout << "A AND B before allgather" <<  my_rank << "\n";
    //     A.print_matrix();
    //     B.print_matrix();
    //     std::cout << "==================================================================================\n";

    //     std::cout << std::endl;
    // }

    // if (my_rank == 0) {
    //     A.print_matrix();
    // }

    // std::cout << "REDISTRIBUTION COMM:  " << my_rank << " " << A_x_real_range.first * A_y_chunk_size + A_y_real_range.first << " "<< A_x_real_range.first << " " << A_y_real_range.first << " " << A_y_chunk_size << std::endl;
    

    if (c > 1) {
        MPI_Comm redistribution_comm;
        MPI_Comm_split(global_comm, A_x_real_range.first * m + A_y_real_range.first, 0, &redistribution_comm);

        double* matrix_recv = (double*) malloc(sizeof(double) * max_chunk_size * max_chunk_size * c);
        MPI_Allreduce(A.data, matrix_recv, max_chunk_size * max_chunk_size, MPI_DOUBLE, MPI_SUM, redistribution_comm);
        free(A.data);
        A.data = matrix_recv;
    }

    //free(matrix_recv);

    

    // if (my_rank == 7) {
    //     std::cout << "==================================================================================\n";
    //     std::cout << "A AND B AFTER ALLGATHER" <<  my_rank << "\n";
    //     A.print_matrix();
    //     B.print_matrix();
    //     std::cout << "==================================================================================\n";

    //     std::cout << std::endl;
    // }


    MatrixChunk C(B_x_chunk_size, A_y_chunk_size);

    //TODO: redistribution
    cannon(p_m, cannon_comm_grid, A, B, C, coords[0], coords[1]);


    // if (my_rank == 3) {
    //     std::cout << "-----------------------------------------------------------------------------------\n";
    //     std::cout << "C AFTER MULTIPLICATION" << "\n";
    //     std::cout << "RANK: " << my_rank << "\n"; 
    //     std::cout << "offsets " << my_x_offset << " " << my_y_offset << "\n";
    //     C.print_matrix();
    //     std::cout << "-----------------------------------------------------------------------------------\n" << std::endl;
    // }

    //TODO: zebranie wszystkiego do 0 k-groupy
    if (p_k > 1){
        MPI_Comm reduction_comm;
        MPI_Comm_split(global_comm, y_coord_in_group * (m / p_m) + x_coord_in_group, 0, &reduction_comm);

        double* reduced_C = (double*) malloc(C.x_size * C.y_size * sizeof(double));
        MPI_Allreduce(C.data, reduced_C, C.x_size * C.y_size, MPI_DOUBLE, MPI_SUM, reduction_comm);
        free(C.data);
        C.data = reduced_C;
    }

    // if (my_rank == 7) {
    //     std::cout << "-----------------------------------------------------------------------------------\n";
    //     std::cout << "C AFTER REDUCTION" << "\n";
    //     std::cout << "RANK: " << my_rank << "\n"; 
    //     // std::cout << "offsets " << my_x_offset << " " << my_y_offset << "\n";
    //     C.print_matrix();
    //     std::cout << "-----------------------------------------------------------------------------------\n" << std::endl;
    // }

    int C_x_chunk_size, C_y_chunk_size;
    C_x_chunk_size = divide_ceil(n, p_n);
    C_y_chunk_size = divide_ceil(m, p_m);
    C_x_range = std::make_pair(x_coord_in_group * C_x_chunk_size, (x_coord_in_group + 1) * C_x_chunk_size);
    C_y_range = std::make_pair(y_coord_in_group * C_y_chunk_size, (y_coord_in_group + 1) * C_y_chunk_size);

    

    if (print_matrix) {
        if (my_rank == 0) {
            if (transpose) {
                double* recv_col = (double*) malloc(C_y_chunk_size * sizeof(double));
                MPI_Status s;
                std::cout << n << " " << m << std::endl; //TODO: C_x_chunk size ograniczyć przy prinotwaniu
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

                //free(recv_col);
            }
            else {
                double* recv_row = (double*) malloc(C_x_chunk_size * sizeof(double));
                MPI_Status s;
                std::cout << m << " " << n << std::endl;
                //std::cout << C_x_chunk_size << " " << C_y_chunk_size << std::endl;
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

                //free(recv_row);
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

                //free(column_buf);
            }
            else {
                for (int row = 0; row < C_y_chunk_size; row++) {
                    MPI_Send(C.data + row * C_x_chunk_size, C_x_chunk_size, MPI_DOUBLE, 0, 0, global_comm);
                }
            }
        }
    }
    else {
        int num_ge = 0;

        if (my_k_group == 0){

            for (int row = 0; row < C.y_size; row++) {
                for (int column = 0; column < C.x_size; column++) {
                    if (C.data[row * C.x_size + column] >= ge_value) {
                        num_ge++;
                    }
                }
            }
        }

        // std::cout << "GE: " << my_rank << " " << num_ge << std::endl;


        int result;
        MPI_Reduce(&num_ge, &result, 1, MPI_INT, MPI_SUM, 0, global_comm);

        if (my_rank == 0)
            std::cout << result << std::endl;
    }
}