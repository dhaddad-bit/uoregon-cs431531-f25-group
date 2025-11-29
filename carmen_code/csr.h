#ifndef CSR_H
#define CSR_H

  #include "mmio.h"
  #include "common.h"
  #include "coo.h"

#ifdef __cplusplus
extern "C" {
#endif



void convert_coo_to_csr(int* row_ind, int* col_ind, double* val, 
		int m, int n, int nnz, unsigned int** csr_row_ptr, 
		unsigned int** csr_col_ind, double** csr_vals);


void csr_spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
		double* csr_vals, int m, int n, int nnz, double* vector_x, double *res);




#ifdef __cplusplus
}
#endif

#endif
