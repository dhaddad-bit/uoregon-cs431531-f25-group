#ifndef MAIN_H
#define MAIN_H
extern "C" {
  #include "mmio.h"
  #include "common.h"
}

void usage(int argc, char** argv);
void print_matrix_info(char* fileName, MM_typecode matcode, 
                       int m, int n, int nnz);
void check_mm_ret(int ret);
void read_info(char* fileName, int* is_sym);
void convert_coo_to_csr(int* row_ind, int* col_ind, P_TYPE* val, 
                        int m, int n, int nnz,
                        unsigned int** csr_row_ptr, unsigned int** csr_col_ind,
                        P_TYPE** csr_vals);
void read_vector(char* fileName, P_TYPE** vector, int* vecSize);
// added
void spmv_coo_cpu(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
              P_TYPE* csr_vals, int m, int n, int nnz, 
              P_TYPE* vector_x, P_TYPE *res, omp_lock_t* writelock);

void spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
          P_TYPE* csr_vals, int m, int n, int nnz, 
          P_TYPE* vector_x, P_TYPE *res);
// next two added
void spmv_coo_ser_cpu(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
                  P_TYPE* csr_vals, int m, int n, int nnz, 
                  P_TYPE* vector_x, P_TYPE *res);
void spmv_ser_cpu(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
              P_TYPE* csr_vals, int m, int n, int nnz, 
              P_TYPE* vector_x, P_TYPE *res);         
void store_result(char *fileName, P_TYPE* res, int m);
void print_time(P_TYPE timer[]);
void expand_symmetry(int m, int n, int* nnz_, int** row_ind, int** col_ind, 
                     P_TYPE** val);
P_TYPE ddot(const int n, P_TYPE* x, const int incx, P_TYPE* y, const int incy);
P_TYPE dnrm2(const int n, P_TYPE* x, const int incx);
void convert_csr_to_ell(unsigned int* csr_row_ptr, unsigned int* csr_col_ind,
                        P_TYPE* csr_vals, int m, int n, int nnz, 
                        unsigned int** ell_col_ind, P_TYPE** ell_vals, 
                        int* n_new);

void init_locks(omp_lock_t** locks, int m);
void destroy_locks(omp_lock_t* locks, int m);

// Template MUST be outside extern "C"
template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in);

#endif