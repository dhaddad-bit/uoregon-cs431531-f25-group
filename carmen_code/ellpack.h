#ifndef ELLPACK_H
#define ELLPACK_H

extern "C" {
  #include "mmio.h"
  #include "common.h"
}

void convert_csr_to_ell(unsigned int* csr_row_ptr, unsigned int* csr_col_ind,
                        double* csr_vals, int m, int n, int nnz, 
                        unsigned int** ell_col_ind, double** ell_vals, 
                        int* n_new);

                        
#endif
