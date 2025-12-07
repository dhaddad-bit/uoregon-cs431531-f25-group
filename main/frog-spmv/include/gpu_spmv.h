#ifndef COO_CSR_H
#define COO_CSR_H

#include "common.h"

void scalar_csr(unsigned int* col_ind, unsigned int* row_ptr, P_TYPE* vals, 
                    int m, int n, int nnz, P_TYPE* x, P_TYPE* b, float* time_ms);

void vector_csr(unsigned int* col_ind, unsigned int* row_ptr, P_TYPE* vals, 
                    int m, int n, int nnz, P_TYPE* x, P_TYPE* b, float* time_ms);

void scalar_coo(unsigned int* col_ind, unsigned int* row_ind, P_TYPE* vals, 
                    int m, int n, int nnz, P_TYPE* x, P_TYPE* b, float* time_ms);

void vector_coo(unsigned int *col_ind, unsigned int *row_ind, P_TYPE *vals,
                    int m, int n, int nnz, P_TYPE *x, P_TYPE *b, float* time_ms);

// void get_result_gpu(P_TYPE* dev_b, P_TYPE* b, int m);

// void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
//                       P_TYPE* vals, int m, int n, int nnz, P_TYPE* x, 
//                       unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
//                       P_TYPE** dev_vals, P_TYPE** dev_x, P_TYPE** dev_b);

void allocate_coo_gpu(unsigned int* row_ind, unsigned int* col_ind, 
                      P_TYPE* vals, int m, int n, int nnz, P_TYPE* x, 
                      unsigned int** dev_row_ind, unsigned int** dev_col_ind,
                      P_TYPE** dev_vals, P_TYPE** dev_x, P_TYPE** dev_b);

#endif