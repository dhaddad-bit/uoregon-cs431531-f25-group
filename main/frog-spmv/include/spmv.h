#ifndef SPMV_H
#define SPMV_H

// Make header C-compatible for C++ compilers
#ifdef __cplusplus
extern "C" {
#endif

#include "common.h"

void spmv_gpu_ell(unsigned int* col_ind, P_TYPE* vals, int m, int n, int nnz, 
                  P_TYPE* x, P_TYPE* b, unsigned int threads, float* time_ms);

void allocate_ell_gpu(unsigned int* col_ind, P_TYPE* vals, int m, int n, 
                      int nnz, P_TYPE* x, unsigned int** dev_col_ind, 
                      P_TYPE** dev_vals, P_TYPE** dev_x, P_TYPE** dev_b);

void spmv_gpu_2(unsigned int* row_ptr, unsigned int* col_ind, P_TYPE* vals,
                int m, int n, int nnz, P_TYPE* x, P_TYPE* b, unsigned int threads, float* time_ms);

void spmv_gpu(unsigned int* row_ptr, unsigned int* col_ind, P_TYPE* vals,
              int m, int n, int nnz, P_TYPE* x, P_TYPE* b, unsigned int threads, float* time_ms);

void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
                      P_TYPE* vals, int m, int n, int nnz, P_TYPE* x, 
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      P_TYPE** dev_vals, P_TYPE** dev_x, P_TYPE** dev_b);

void get_result_gpu(P_TYPE* dev_b, P_TYPE* b, int m);

// Fixed: Removed the trailing comma after d_y
void spmv_gpu_sellc(int m, int num_slices, int SLICE_THICKNESS,
                    unsigned int* d_slice_ptr, unsigned int* d_col_ind, 
                    P_TYPE* d_vals, P_TYPE* d_x, P_TYPE* d_y, float* time_ms);

void cusparse_csr(int m, int n, int nnz,
                        unsigned int *row_ptr, unsigned int *col_ind, P_TYPE *vals,
                        P_TYPE *d_x, P_TYPE *d_b, float *time_ms);
void cusparse_coo(int m, int n, int nnz,                                                                  
                  unsigned int *row_ind, 
                  unsigned int *col_ind, P_TYPE *vals,
                  P_TYPE *d_x, P_TYPE *d_b, float *time_ms);

#ifdef __cplusplus
} // extern "C"
#endif

#endif