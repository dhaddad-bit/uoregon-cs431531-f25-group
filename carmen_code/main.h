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


void read_vector(char* fileName, double** vector, int* vecSize);
// added


void store_result(char *fileName, double* res, int m);

void print_time(double timer[]);

void expand_symmetry(int m, int n, int* nnz_, int** row_ind, int** col_ind, 
                     double** val);

double ddot(const int n, double* x, const int incx, double* y, const int incy);

double dnrm2(const int n, double* x, const int incx);

void init_locks(omp_lock_t** locks, int m);

void destroy_locks(omp_lock_t* locks, int m);

#endif