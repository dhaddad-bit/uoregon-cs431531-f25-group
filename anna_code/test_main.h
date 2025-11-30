void read_vector(const char *file, double **vector, int *size);

void read_matrix(const char *file, int *num_rows, int *num_cols, int *num_nnz, 
    unsigned int **rows, unsigned int **cols, double **values);

void convert_coo_to_csr(unsigned int *row_ind, unsigned int *col_ind, double *val, 
                unsigned int **csr_row_ptr, unsigned int **csr_col_ind, double **csr_vals,
                int m, int n, int nnz);

void spmv_csr(unsigned int *row_ptr, unsigned int *col_ind, double *vals,
     int m, int n, int nnz, double *vector, double *result);

void store_result(const char *file, double *result, int m);
