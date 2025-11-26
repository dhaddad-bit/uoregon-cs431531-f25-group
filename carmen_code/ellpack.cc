#include <stdio.h>

#include "csr.cc"



void convert_csr_to_ell(unsigned int* csr_row_ptr, unsigned int* csr_col_ind,
                        double* csr_vals, int m, int n, int nnz, 
                        unsigned int** ell_col_ind, double** ell_vals, 
                        int* n_new)
{
    // --- Find the max number of non-zeros per row (max_row_length) --- 
    int max_row_length = 0;
    for (int i=0; i<m; i++) {
        int row_length =csr_row_ptr[i+1] - csr_row_ptr[i];
        if (row_length > max_row_length) {
            max_row_length = row_length;
        }
    }
    *n_new = max_row_length; // from now on this is the length of our rows
    int ell_nnz_padded = m * max_row_length;
    
    // --- Allocate memory for ELL arrays on the host (CPU) --- 
    *ell_col_ind = (unsigned int*) malloc(sizeof(unsigned int) * ell_nnz_padded);
    assert(*ell_col_ind);
    *ell_vals = (double*) malloc(sizeof(double) * ell_nnz_padded);
    assert(*ell_vals);

    // --- Initialize ELL arrays to zero for padding use (-1) for col/row and 0 for vals --- 
    // Can this be done with calloc or memset? Parallelizable with omp? TODO
    for (int i=0; i<ell_nnz_padded; i++) {
        (*ell_col_ind)[i] = -1; // UNSIGNED INT ISSUE?;
        (*ell_vals)[i] = 0.0;
    }
        
    // --- Fill in ELL arrays from CSR arrays --- 
    // (See slides and paper for details)
    #pragma omp parallel for schedule(static)
    for (int i=0; i<m; ++i) { // "for each row"
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i+1];
        int c = 0;
        // Copy data from CSR arrays to ELL arrays
        for (int j=row_start; j<row_end; ++j) {
            int ell_dest_index = i*max_row_length + c; // destination_index = row_index * max_row_length + col_index
            (*ell_col_ind)[ell_dest_index] = csr_col_ind[j];
            (*ell_vals)[ell_dest_index] = csr_vals[j];
            c++;
        }
        // Moved padding initialization above loop to -1/0.0 values.
    }
        // COMPLETE THIS FUNCTION
}