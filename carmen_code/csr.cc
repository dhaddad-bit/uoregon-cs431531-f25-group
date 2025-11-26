#include <assert.h>
#include <cstring>

#include "csr.h"
#include "coo.h"


/* SpMV function for CSR stored sparse matrix
 */
void csr_spmv_ser_cpu(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
              double* csr_vals, int m, int n, int nnz, 
              double* vector_x, double *res)
{
    // Serial vereion for SpMV CSR format
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = csr_row_ptr[i]; j<csr_row_ptr[i+1]; j++) {
            sum += csr_vals[j] * vector_x[csr_col_ind[j]];
        }
        res[i] = sum;
    }
}


/* SpMV function for CSR stored sparse matrix
 */
void csr_spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
          double* csr_vals, int m, int n, int nnz, 
          double* vector_x, double *res)
{
    // first initialize res to 0
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < m; i++) {
        res[i] = 0.0;
    }

    // calculate spmv
    #pragma omp parallel for schedule(static)
    for(unsigned int i = 0; i < m; i++) {
        unsigned int row_begin = csr_row_ptr[i];
        unsigned int row_end = csr_row_ptr[i + 1];
        for(unsigned int j = row_begin; j < row_end; j++) {
            res[i] += csr_vals[j] * vector_x[csr_col_ind[j]]; 
        }
    }
}

void convert_coo_to_csr(int* row_ind, int* col_ind, double* val, 
                        int m, int n, int nnz,
                        unsigned int** csr_row_ptr, unsigned int** csr_col_ind,
                        double** csr_vals)

{
    // Temporary pointers
    unsigned int* row_ptr_;
    unsigned int* col_ind_;
    double* vals_;
    
    // We now how large the data structures should be
    // csr_row_ptr -> m + 1
    // csr_col_ind -> nnz
    // csr_vals    -> nnz
    row_ptr_ = (unsigned int*) malloc(sizeof(unsigned int) * (m + 1)); 
    assert(row_ptr_);
    col_ind_ = (unsigned int*) malloc(sizeof(unsigned int) * nnz);
    assert(col_ind_);
    vals_ = (double*) malloc(sizeof(double) * nnz);
    assert(vals_);

    // Now determine how many non-zero elements are in each row
    // Use a histogram to do this
    unsigned int* buckets = (unsigned int*) malloc(sizeof(unsigned int) * m);
    assert(buckets);
    memset(buckets, 0, sizeof(unsigned int) * m);

    for(unsigned int i = 0; i < nnz; i++) {
        // row_ind[i] - 1 because index in mtx format starts from 1 (not 0)
        buckets[row_ind[i] - 1]++;
    }

    // Now use a cumulative sum to determine the starting position of each
    // row in csr_col_ind and csr_vals - this information is also what is
    // stored in csr_row_ptr
    for(unsigned int i = 1; i < m; i++) {
        buckets[i] = buckets[i] + buckets[i - 1];
    }
    // Copy this to csr_row_ptr
    row_ptr_[0] = 0; 
    for(unsigned int i = 0; i < m; i++) {
        row_ptr_[i + 1] = buckets[i];
    }

    // We can use row_ptr_ to copy the column indices and vals to the 
    // correct positions in csr_col_ind and csr_vals
    unsigned int* tmp_row_ptr = (unsigned int*) malloc(sizeof(unsigned int) * 
                                                       m);
    assert(tmp_row_ptr);
    memcpy(tmp_row_ptr, row_ptr_, sizeof(unsigned int) * m);

    // Now go through each non-zero and copy it to its appropriate position
    for(unsigned int i = 0; i < nnz; i++)  {
        col_ind_[tmp_row_ptr[row_ind[i] - 1]] = col_ind[i] - 1;
        vals_[tmp_row_ptr[row_ind[i] - 1]] = val[i];
        tmp_row_ptr[row_ind[i] - 1]++;
    }

    // Copy the memory address to the input parameters
    *csr_row_ptr = row_ptr_;
    *csr_col_ind = col_ind_;
    *csr_vals = vals_;

    // Free memory
    free(tmp_row_ptr);
    free(buckets);
}
