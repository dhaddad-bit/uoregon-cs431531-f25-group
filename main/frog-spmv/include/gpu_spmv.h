#ifndef COO_CSR_H
#define COO_CSR_H

void scalar_csr(unsigned int* col_ind, unsigned int* row_ptr, double* vals, 
                    int m, int n, int nnz, double* x, double* b, float* time_ms);

void vector_csr(unsigned int* col_ind, unsigned int* row_ptr, double* vals, 
                    int m, int n, int nnz, double* x, double* b, float* time_ms);

void scalar_coo(unsigned int* col_ind, unsigned int* row_ind, double* vals, 
                    int m, int n, int nnz, double* x, double* b, float* time_ms);

void vector_coo(unsigned int *col_ind, unsigned int *row_ind, double *vals,
                    int m, int n, int nnz, double *x, double *b, float* time_ms);

// void get_result_gpu(double* dev_b, double* b, int m);

// void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
//                       double* vals, int m, int n, int nnz, double* x, 
//                       unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
//                       double** dev_vals, double** dev_x, double** dev_b);

void allocate_coo_gpu(unsigned int* row_ind, unsigned int* col_ind, 
                      double* vals, int m, int n, int nnz, double* x, 
                      unsigned int** dev_row_ind, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b);

#endif