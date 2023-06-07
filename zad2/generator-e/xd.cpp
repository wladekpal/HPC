c = p_n / p_m;
        x_dim_cannon_group = p_n / c;
        y_dim_cannon_group = p_m;

        assert (x_dim_cannon_group == y_dim_cannon_group);

        cannon_size = x_dim_cannon_group * y_dim_cannon_group;
        my_cannon_group = my_id_in_group / cannon_size;
        my_id_in_cannon = my_id_in_group % cannon_size;

        x_group_offset = my_k_group * (n / p_n);
        y_group_offset = 0;
        
        A_x_chunk_size = c * (k / p_k) / p_n;
        A_y_chunk_size = m / p_m;

        B_x_chunk_size = n / p_n;
        B_y_chunk_size = (k / p_k) / p_m;

        x_coord_in_group = my_id_in_cannon / y_dim_cannon_group;
        y_coord_in_group = my_id_in_cannon % y_dim_cannon_group;

        x_coord_in_cannon = x_coord_in_group % c;
        y_coord_in_cannon = y_coord_in_group;

        replication_offset = 0;//my_cannon_group * A_x_chunk_size;

        A_x_gen_range = std::make_pair(my_cannon_group * (k / c) + my_k_group * (k / p_k), (my_cannon_group + 1) * (k / c) + my_k_group * (k / p_k));
        A_y_gen_range = std::make_pair(y_coord_in_cannon * A_y_chunk_size + y_group_offset, (y_coord_in_cannon + 1) * A_y_chunk_size + y_group_offset);

        A_x_chunk_size = A_x_chunk_size * c; // This has to be expanded, to store replicated block
        A_x_real_range = std::make_pair( my_k_group * (k / p_k),  (my_k_group + 1) * (k / p_k));
        A_y_real_range = A_y_gen_range;

        B_x_gen_range = std::make_pair(x_coord_in_k_group * B_x_chunk_size + x_group_offset, (x_coord_in_k_group + 1) * B_x_chunk_size + x_group_offset);
        B_y_gen_range = std::make_pair(y_coord_in_k_group * B_y_chunk_size + y_group_offset, (y_coord_in_k_group + 1) * B_y_chunk_size + y_group_offset);

        B_x_real_range = B_x_gen_range;
        B_y_real_range = B_y_gen_range;