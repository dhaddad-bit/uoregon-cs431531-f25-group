#ifndef CSR5_H
#define CSR5_H

extern "C" {
  #include "mmio.h"
  #include "common.h"
  #include "ellpack.h"
}

//loading csr5


void convert_to_csr5(int* row_ind, int* col_ind, double* val, 
		int m, int n, int nnz, unsigned int** csr_row_ptr, 
		unsigned int** csr_col_ind, double** csr_vals);


//void csr5_spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
//		double* csr_vals, int m, int n, int nnz, double* vector_x, double *res);




#ifdef __cplusplus
}
#endif

#endif



//this file will define kernel funtionality

//also do calculation y = Ax

#endif
