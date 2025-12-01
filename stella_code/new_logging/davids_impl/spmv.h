#ifndef SPMV_H
#define SPMV_H

// Make header C-compatible for C++ compilers
#ifdef __cplusplus
extern "C" {
#endif

#include "common.h"

// Note: Removed "= 64" default arguments for C-compatibility
// Note: Time for DATA GETTING THAT DATAAAAA: 
    // I will be adding float* time_ms to all GPU launchers

void spmv_gpu_ell(unsigned int* col_ind, double* vals, int m, int n, int nnz, 
                  double* x, double* b, unsigned int threads, float* time_ms);

void allocate_ell_gpu(unsigned int* col_ind, double* vals, int m, int n, 
                      int nnz, double* x, unsigned int** dev_col_ind, 
                      double** dev_vals, double** dev_x, double** dev_b);

void spmv_gpu_2(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
                int m, int n, int nnz, double* x, double* b, unsigned int threads, float* time_ms);

void spmv_gpu(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
              int m, int n, int nnz, double* x, double* b, unsigned int threads, float* time_ms);

void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
                      double* vals, int m, int n, int nnz, double* x, 
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b);

void get_result_gpu(double* dev_b, double* b, int m);

// Fixed: Removed the trailing comma after d_y
void spmv_gpu_sellc(int m, int num_slices, int SLICE_THICKNESS,
                    unsigned int* d_slice_ptr, unsigned int* d_col_ind, 
                    double* d_vals, double* d_x, double* d_y, float* time_ms);

#ifdef __cplusplus
} // extern "C"
#endif

// Template MUST be outside extern "C"
template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in);

#endif // SPMV_H

