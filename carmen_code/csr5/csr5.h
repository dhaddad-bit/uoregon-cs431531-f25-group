#ifndef CSR5_H
#define CSR5_H

#include <stdint.h> // for uint32_t, uint64_t
// --- STPUID WRAPPER THING I DONT UNDERSTAND ---
#ifdef __cplusplus
extern "C" {
#endif
// --- STPUID WRAPPER THING I DONT UNDERSTAND ---

#define CSR5_OMEGA 32
#define CSR5_SIGMA 16
// Host-side conversion function
void convert_csr_to_csr5(
    int m, int n, int nnz,
    const unsigned int* h_row_ptr,
    const unsigned int* h_col_idx,
    const double* h_val,
    // Outputs
    int* num_tiles,
    double** h_csr5_val,
    int** h_csr5_col_idx,
    int** h_csr5_row_idx, // Row versioning of CSR5 to test what's wrong with my computation logic,, i rushed it at 2am and it doesn't work
    int** h_csr5_tile_ptr,
    uint32_t** h_csr5_tile_desc
);

// Device-side SpMV function
void spmv_gpu_csr5(
    int m,
    int num_tiles,
    double* d_val,
    int* d_col_idx,
    int* d_row_idx, // Row versioning of CSR5
    int* d_tile_ptr,
    uint32_t* d_tile_desc,
    double* d_x,
    double* d_y,
    float* time_ms
);


void convert_csr_to_csr5_gpu(
    //Inputs
    int m, int n, int nnz,
    const unsigned int* og_row_ptr,
    const unsigned int* og_col_idx,
    const double* og_val,
    int *omega, int *sigma,
    // Outputs
    int* num_tiles,
    double** h_csr5_val,
    int** h_csr5_col_idx,
    int** h_csr5_row_idx, 
    int** h_csr5_tile_ptr,
    uint32_t** h_csr5_tile_desc
);



// --- STPUID WRAPPER THING I DONT UNDERSTAND ---
#ifdef __cplusplus
}
#endif

#endif 
// --- STPUID WRAPPER THING I DONT UNDERSTAND ---
