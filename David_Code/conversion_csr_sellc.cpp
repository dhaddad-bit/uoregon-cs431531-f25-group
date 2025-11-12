// This will run on the CPU
#include <vector>
// #include <numeric>
// #include <algorithm>
#include <iostream>
#include <cuda_runtime.h>
#include <cassert>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*
 * Converts a CSR matrix to CELL-C format on the CPU.
 * Parameters:
    * m: number of rows
    * csr_row_ptr: CSR row pointer array
    * csr_col_ind: CSR column indices array
    * csr_vals: CSR values array
    * SLICE_THICKNESS: size of each cell sclice
    * Output:
        * m_padded: number of rows after padding
        * total_nnz: number of elements in padded array (CELL-C format)
        * sell_slice_ptr: SELL-C slice pointner array
        * sell_col_ind: SELL-C colunmn indices array
        * sell_vals: SELL-C values array
 */
void convert_csr_to_sell_c(int m, const unsigned int* csr_row_ptr, const unsigned int* csr_col_ind, const double* csr_vals, int SLICE_THICKNESS, int* m_padded, int* total_nnz, unsigned int** sell_slice_ptr, unsigned int** sell_col_ind, double** sell_vals)
{
    int num_slices = (m + SLICE_THICKNESS -1) / SLICE_THICKNESS;
    *m_padded = num_slices * SLICE_THICKNESS;
    std::vector<unsigned int> slice_lengths(num_slices);
    for (int s = 0; s < num_slices; s++) {
        unsigned int max_length = 0;
        for (int i=0; i<SLICE_THICKNESS; i++) {
            int row = s * SLICE_THICKNESS + i;
            if (row < m) { // we are within original matrix (unpadded bounds)
                unsigned int row_length = csr_row_ptr[row+1] - csr_row_ptr[row];
                max_length = std::max(max_length, row_length);
    
            }
        }
        slice_lengths[s] = max_length;
    }
    // Prefix sum on slice lengths to get the slice_ptr array
    *sell_slice_ptr = (unsigned int*)malloc((sizeof(unsigned int) * (num_slices + 1)));
    assert(*sell_slice_ptr);

    unsigned int padded_nnz = 0;
    for (int s=0; s<num_slices, s++) {
        (*sell_slice_ptr)[s] = padded_nnz;
        padded_nnz += slice_lengths[s] * SLICE_THICKNESS;
    }
    (*sell_slice_ptr)[num_slices] = padded_nnz;
    *total_nnz = padded_nnz;
    // Allocate and fill arrays for SELL-C format
    *sell_col_ind = (unsigned int*)calloc(*total_nnz, sizeof(unsigned int));
    *sell_vals = (double*)calloc(*total_nnz, sizeof(double));
    assert(*sell_col_ind);
    assert(*sell_vals);

    // Each slice gets iterated over
    for (int s=0; s<num_slices; s++) {
        unsigned int slice_start = (*sell_slice_ptr)[s];
        unsigned int slice_max_length = slice_lengths[s];

        // Iterate threads/rows within the slice
        for (int i=0; i<SLICE_THICKNESS; i++) {
            int row = s * SLICE_THICKNESS + i;
            if (row < m) { // non padded rows
                unsigned int csr_start = csr_row_ptr[row];
                unsigned int csr_end = csr_row_ptr[row+1];
                unsigned int row_length = csr_end - csr_start;
            // Copy the data from CSR to SELL-C
                for (unsigned int j=0; j<row_length; j++) {
                    unsigned int sell_index = slice_start + j * SLICE_THICKNESS + i;
                    (*sell_col_ind)[sell_index] = csr_col_ind[csr_start + j];
                    (*sell_vals)[sell_index] = csr_vals[csr_start + j];
                }
            }
            // Padding is already zeroed out by calloc
        }
    }
}