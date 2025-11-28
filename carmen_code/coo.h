#ifndef COO_H
#define COO_H

#pragma once
#include <stdio.h>
#include <cstring>
#include <string.h>
#include <omp.h>
#include <assert.h>


extern "C" {
  #include "mmio.h"
  #include "common.h"
}




void spmv_coo_cpu(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
              double* csr_vals, int m, int n, int nnz, 
              double* vector_x, double *res, omp_lock_t* writelock);



// next two added
void spmv_coo_ser_cpu(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
                  double* csr_vals, int m, int n, int nnz, 
                  double* vector_x, double *res);

void spmv_ser_cpu(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
              double* csr_vals, int m, int n, int nnz, 
              double* vector_x, double *res);       


#endif
