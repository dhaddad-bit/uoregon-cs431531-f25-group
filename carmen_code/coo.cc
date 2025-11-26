#include "coo.h"


// Code from the third homework for cpu calculation of CSR and COO
void spmv_coo_cpu(unsigned int* row_ind, unsigned int* col_ind, double* vals, 
              int m, int n, int nnz, double* vector_x, double *res, 
              omp_lock_t* writelock)
{
    #pragma omp parallel for
    for (int i=0; i<m; i++) {
        res[i] = 0.0;
    }

    // SpMV Calculation (parallelizable)
    // each thread processes a section of non-zero elts
    #pragma omp parallel for
    for (int i=0; i<nnz; i++) {
        // 1-based to 0-based errors my gosh
        int row = row_ind[i]-1;
        int col = col_ind[i]-1;
        // MUST PREVENT RACE CONDITION updating res[row]
        // Attomic version first draft (less complex/less efficient)
        // #pragma omp atomic
        // res[row] += vals[i] * vector_x[col];
        omp_set_lock(&(writelock[row]));
        res[row] += vals[i] * vector_x[col];
        omp_unset_lock(&(writelock[row]));
    }
}

void spmv_coo_ser_cpu(unsigned int* row_ind, unsigned int* col_ind, double* vals, 
                  int m, int n, int nnz, double* vector_x, double *res)
{
    // Serial version for COO SpMV
    // Initialize result vector to zero
    for (int i=0; i<m; i++) {
        res[i] = 0.0;
    }
    // Cacluate SpMV
    for (int i=0; i<nnz; i++) {
        int row = row_ind[i]-1;
        int col = col_ind[i]-1;
        res[row] += vals[i] * vector_x[col];
    }
}
