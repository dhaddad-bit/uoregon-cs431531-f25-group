#ifndef COO_CSR_H
#define COO_CSR_H

void scalar_csr(unsigned int* col_ind, unsigned int* row_ptr, double* vals, 
                    int m, int n, int nnz, double* x, double* b);

void vector_csr(unsigned int* col_ind, unsigned int* row_ptr, double* vals, 
                    int m, int n, int nnz, double* x, double* b);

void scalar_coo(unsigned int* col_ind, unsigned int* row_ind, double* vals, 
                    int m, int n, int nnz, double* x, double* b);

#endif