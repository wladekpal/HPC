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

            for (int row = 0; row < size; row++) {
                for (int column = 0; column < size; column++) {
                    if (is_in_range(column + x_offset, row + y_offset, x_gen_range, y_gen_range)) {
                        this->data[row * size + column] = generate_double(seed, row + y_offset, column + x_offset);
                    }
                }
            }
                
        }

        ~MatrixChunk() {
            //free((void*)this->data);
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


void print_row(double* data, int size) {
    for (int i = 0 ; i < size; i++)
        std::cout << data[i] << " ";
}


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

                // if (my_rank == 0)
                //     std::cout << "Proc: " << my_rank << " i:" << i << " j:" << j << " k:" << k  << " " << A.data[i * n + k] << " " << B.data[k * n + j]<< std::endl;
            }
        }
    }
}



void cannon(int grid_size, MPI_Comm comm, MatrixChunk &A, MatrixChunk &B, MatrixChunk &C, int skew_vertical, int skew_horizontal) {
    MPI_Request send_horizontal, send_vertical, recv_horizontal, recv_vertical;
    MPI_Status s;

    int left, right, down, up, skew_left, skew_right, skew_up, skew_down;

    MPI_Cart_shift(comm, 0, 1, &left, &right);
    MPI_Cart_shift(comm, 1, 1, &up, &down);
    MPI_Cart_shift(comm, 0, skew_horizontal, &skew_left, &skew_right);
    MPI_Cart_shift(comm, 1, skew_vertical, &skew_up, &skew_down);


    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    

    if (grid_size == 1) {
        multiply_add(A, B, C);
        return;
    }

    
    if (skew_horizontal > 0) {
        double* buff = (double*) malloc(A.size * A.size * sizeof(double));
        //std::cout << "BEFORE SKEW" << my_rank << std::endl;
        MPI_Sendrecv(A.data, A.size * A.size, MPI_DOUBLE, skew_left, 0, buff, A.size * A.size, MPI_DOUBLE, skew_right, 0, comm, &s);
        free(A.data);
        A.data = buff;
        // std::cout << "AFTER SKEW1" << my_rank << std::endl;

    }

    if (skew_vertical > 0) {
        double* buff = (double*) malloc(B.size * B.size * sizeof(double));
        MPI_Sendrecv(B.data, B.size * B.size, MPI_DOUBLE, skew_up, 0, buff, B.size * B.size, MPI_DOUBLE, skew_down, 0, comm, &s);
        //std::cout << "AFTER SKEW" << my_rank << std::endl;
        free(B.data);
        B.data = buff;
        // std::cout << "AFTER SKEW2" << my_rank << std::endl;
    }

    if (my_rank >= 0){
        // std::cout << "Before" << my_rank << " skew " << skew << "\n";
        // A.print_matrix();
        // B.print_matrix();
        // C.print_matrix();
        // std::cout << "Neighb" << my_rank << " " << left << " " << right << " " << up << " " << down << std::endl;
    }

    for (int i = 0 ; i < grid_size; i++) {


        // send_horizontal = send_async_matrix(comm, left, A);
        // send_vertical = send_async_matrix(comm, up, B);
        // MPI_Send(A, A.size * A.size, MPI_DOUBLE, left, 0, comm);
        // MPI_Send(B, B.size * B.szie, MPI_DOUBLE, up, 0, comm);

        multiply_add(A, B, C);


        // MPI_Wait(&send_horizontal, &s);
        // std::cout << "SENT HORIZONTAL" << std::endl;
        // MPI_Wait(&send_vertical, &s);

        // std::cout << "SENT" << std::endl;

        // recv_horizontal = recv_async_matrix(comm, right, A);
        // recv_vertical = recv_async_matrix(comm, down, B);


        // MPI_Wait(&recv_horizontal, &s);
        // MPI_Wait(&recv_vertical, &s);

        double* temp_A = (double*) malloc(A.size * A.size * sizeof(double));
        double* temp_B = (double*) malloc(B.size * B.size * sizeof(double));

        MPI_Sendrecv(A.data, A.size * A.size, MPI_DOUBLE, left, 0, temp_A, A.size * A.size, MPI_DOUBLE, right, 0, comm, &s);
        MPI_Sendrecv(B.data, B.size * B.size, MPI_DOUBLE, up, 0, temp_B, B.size * B.size, MPI_DOUBLE, down, 0, comm, &s);

        free(A.data);
        free(B.data);
        A.data = temp_A;
        B.data = temp_B;
        // if (my_rank == 0){
        //     std::cout << "After:\n";
        //     A.print_matrix();
        //     B.print_matrix();
        //     C.print_matrix();
        // }
    }
}

void multiply(int n, int m, int k, int seed_A, int seed_B, std::tuple<int, int, int> p_counts, bool print_matrix, double ge_value) {
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
    int my_x_offset, my_y_offset;

    


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
        
        A_x_chunk_size = (k / p_k) / x_dim_cannon_group;//(k / p_k) / p_n;
        A_y_chunk_size = m / p_m;

        
        x_coord_in_group = my_id_in_group / p_m;
        y_coord_in_group = my_id_in_group % p_m;

        if (c > 1)
            x_coord_in_cannon = x_coord_in_group % c;
        else
            x_coord_in_cannon = x_coord_in_group;
        y_coord_in_cannon = y_coord_in_group;

        // std::cout << "XDDD " << my_rank << " " << x_coord_in_group << " " << y_coord_in_group << " -> " << A_x_chunk_size << " " << x_coord_in_group * A_x_chunk_size + x_group_offset << " " << (x_coord_in_group + 1) * A_x_chunk_size + x_group_offset << std::endl;
        A_x_gen_range = std::make_pair(x_coord_in_cannon * A_x_chunk_size + x_group_offset + my_cannon_group * (A_x_chunk_size / c), x_coord_in_cannon * A_x_chunk_size + x_group_offset + (my_cannon_group + 1) * (A_x_chunk_size / c) );
        A_y_gen_range = std::make_pair(y_coord_in_group * A_y_chunk_size + y_group_offset, (y_coord_in_group + 1) * A_y_chunk_size + y_group_offset);

        // A_x_chunk_size = k / x_dim_cannon_group; // This has to be expanded, to store replicated block
        A_x_real_range = std::make_pair(x_coord_in_cannon * A_x_chunk_size + x_group_offset, (x_coord_in_cannon + 1) * A_x_chunk_size + x_group_offset);

        // A_x_real_range = A_x_gen_range;
        // if (A_x_real_range.first % A_x_chunk_size != 0)
        //     A_x_real_range.first = A_x_real_range.first - (A_x_real_range.first % A_x_chunk_size);
        // if (A_x_real_range.second % A_x_chunk_size != 0)
        //     A_x_real_range.second = A_x_real_range.second - (A_x_real_range.second % A_x_chunk_size) + A_x_chunk_size;

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
        assert(false);
    }

    
    MPI_Barrier(MPI_COMM_WORLD); // TODO: wywalić

    // if (my_rank == 0) {
    //     std::cout << "-----------------------------------------------------------------------------------" << std::endl;
    // }

    MPI_Barrier(MPI_COMM_WORLD); // TODO: wywalić

    MPI_Comm cannon_comm, cannon_comm_grid;
    MPI_Comm_split(MPI_COMM_WORLD, my_k_group * c + my_cannon_group, my_id_in_cannon, &cannon_comm);
    std::cout << my_rank << "->>>>>>" << my_k_group * c + my_cannon_group << std::endl;
    MPI_Barrier(MPI_COMM_WORLD); // TODO: wywalić

    int dims[2] = {x_dim_cannon_group, y_dim_cannon_group};
    int periods[2] = {1, 1};
    MPI_Cart_create(cannon_comm, 2, dims, periods, false, &cannon_comm_grid);

    int coords[2];
    MPI_Barrier(MPI_COMM_WORLD); // TODO: wywalić

    MPI_Cart_coords(cannon_comm_grid, my_id_in_cannon, 2, coords);

    //std::cout << "Proc " << my_rank << "| id_in_cannon: " << my_id_in_cannon << "| coords: " << coords[0] << " " << coords[1] << std::endl; 
    std::cout << my_rank << " -----------------------> coords: " << coords[0] << " " << coords[1] << std::endl;
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


    // if (my_rank >= 0) {
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

    MPI_Comm redistribution_comm;
    MPI_Comm_split(MPI_COMM_WORLD, A_x_real_range.first * A_y_chunk_size + A_y_real_range.first, 0, &redistribution_comm); // TODO: tutaj może na podstawie A_x_real_range trzeba to robić, i wtedy będzie dobrze ??

    //std::cout << "REDISTRIBUTION COMM:  " << my_rank << " " << A_x_real_range.first * A_y_chunk_size + A_y_real_range.first << std::endl;

    // TODO: A trzeba zamalocować lepiej, tak, żeby to od razu allgatherem się tam zapisało albo mpi_allreduce
    double* matrix_send = A.data;//(double*) malloc(sizeof(double) * max_chunk_size * max_chunk_size);
    double* matrix_recv = (double*) malloc(sizeof(double) * max_chunk_size * max_chunk_size * c);

    MPI_Barrier(MPI_COMM_WORLD);

    if (c > 1) {
        MPI_Allgather(matrix_send, max_chunk_size * max_chunk_size, MPI_DOUBLE, matrix_recv, max_chunk_size * max_chunk_size, MPI_DOUBLE, redistribution_comm);

        for (int i = 0; i < max_chunk_size * max_chunk_size; i++) {
            double sum = 0;
            for (int j = 0; j < c; j++) { // TODO: tutaj nie p_n
                sum += matrix_recv[max_chunk_size * max_chunk_size * j + i];
               // std::cout << "ADDRESS" << max_chunk_size * max_chunk_size * j + i << "-->>>>> " << matrix_recv[max_chunk_size * max_chunk_size * j + i] << std::endl;
            }
            A.data[i] = sum;
        }
    }

    

    if (my_rank == 1) {
        std::cout << "==================================================================================\n";
        std::cout << "A AND B AFTER ALLGATHER" <<  my_rank << "\n";
        A.print_matrix();
        B.print_matrix();
        std::cout << "==================================================================================\n";

        std::cout << std::endl;
    }


    MatrixChunk C(max_chunk_size);

    //TODO: redistribution
    cannon(x_dim_cannon_group, cannon_comm_grid, A, B, C, coords[0], coords[1]);


    my_x_offset = B_x_real_range.first;
    my_y_offset = A_y_real_range.first;

    if (my_rank == 1) {
        std::cout << "-----------------------------------------------------------------------------------\n";
        std::cout << "C AFTER MULTIPLICATION" << "\n";
        std::cout << "RANK: " << my_rank << "\n"; 
        std::cout << "offsets " << my_x_offset << " " << my_y_offset << "\n";
        C.print_matrix();
        std::cout << "-----------------------------------------------------------------------------------\n" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //TODO: zebranie wszystkiego do 0 k-groupy
    if (p_k > 1){
        double* reduced_C = (double*) malloc(C.size * C.size * sizeof(double));
        MPI_Comm reduction_comm;
        MPI_Comm_split(MPI_COMM_WORLD, y_coord_in_group * (m / p_m) + x_coord_in_group, 0, &reduction_comm);
        MPI_Allreduce(C.data, reduced_C, C.size * C.size, MPI_DOUBLE, MPI_SUM, reduction_comm);
        free(C.data);
        C.data = reduced_C;
    }

    int C_x_chunk_size, C_y_chunk_size;
    C_x_chunk_size = n / p_n;
    C_y_chunk_size = m / p_m;
    C_x_range = std::make_pair(x_coord_in_group * C_x_chunk_size, (x_coord_in_group + 1) * C_x_chunk_size);
    C_y_range = std::make_pair(y_coord_in_group * C_y_chunk_size, (y_coord_in_group + 1) * C_y_chunk_size);

    

    if (print_matrix) {
        double* recv_row = (double*) malloc(C_x_chunk_size * sizeof(double));
        MPI_Status s;
        if (my_rank == 0) {
            std::cout << m << " " << n << std::endl;
            for (int p_row = 0; p_row < p_m; p_row++) {
                for (int row = p_row * C_y_chunk_size; row < (p_row + 1) * C_y_chunk_size; row++) {
                    int row_in_proc = row % C_y_chunk_size;
                    for (int p_col = 0; p_col < p_n; p_col++) {
                        int proc_num = p_col * p_m + p_row;
                       // std::cout << "->>>>>>>>>>>>>>>>>>>>>> proc: " << proc_num << " " << p_row << " " << p_col << std::endl;
                        if (proc_num == 0) {
                            print_row(C.data + row_in_proc * C.size, C_x_chunk_size);
                        }
                        else {
                            MPI_Recv(recv_row, C_x_chunk_size, MPI_DOUBLE, proc_num, 0, MPI_COMM_WORLD, &s);
                            print_row(recv_row, C_x_chunk_size);
                        }
                    }
                    std::cout << std::endl;
                }
                
            }
        }
        else {
            for (int row = 0; row < C_y_chunk_size; row++) {
                MPI_Send(C.data + row * C_x_chunk_size, C_x_chunk_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

}