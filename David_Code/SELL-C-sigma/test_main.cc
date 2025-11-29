#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

// Forward declarations of our new functions (extern "C" to match)
extern "C" void convert_csr_to_sell_c(int m, const unsigned int* csr_row_ptr, const unsigned int* csr_col_ind, const double* csr_vals, int SLICE_THICKNESS, int* m_padded, int* total_nnz, unsigned int** sell_slice_ptr, unsigned int** sell_col_ind, double** sell_vals);

extern "C" void spmv_gpu_sellc(int m, int SLICE_THICKNESS, int num_slices, unsigned int* d_slice_ptr, unsigned int* d_col_ind, double* d_vals, double* d_x, double* d_y);

int main() {
    // 1. Create a tiny dummy matrix (4x4, diagonal) for testing
    //    1 0 0 0
    //    0 2 0 0
    //    0 0 3 0
    //    0 0 0 4
    int m = 4;
    int nnz = 4;
    unsigned int h_row_ptr[] = {0, 1, 2, 3, 4};
    unsigned int h_col_ind[] = {0, 1, 2, 3};
    double h_vals[] = {1.0, 2.0, 3.0, 4.0};
    
    // Input vector x = {1, 1, 1, 1}
    double h_x[] = {1.0, 1.0, 1.0, 1.0};
    // Expected output y = {1, 2, 3, 4}

    // 2. Run Host Conversion
    int SLICE_THICKNESS = 2; // Small slice size for this tiny matrix
    int m_padded, total_nnz_padded;
    unsigned int *h_sell_slice_ptr, *h_sell_col_ind;
    double *h_sell_vals;

    convert_csr_to_sell_c(m, h_row_ptr, h_col_ind, h_vals, SLICE_THICKNESS, 
                          &m_padded, &total_nnz_padded, 
                          &h_sell_slice_ptr, &h_sell_col_ind, &h_sell_vals);

    printf("Conversion Complete.\n");
    printf("Padded NNZ: %d\n", total_nnz_padded);

    // 3. Allocate GPU Memory
    unsigned int *d_slice_ptr, *d_col_ind;
    double *d_vals, *d_x, *d_y;
    int num_slices = m_padded / SLICE_THICKNESS;

    cudaMalloc(&d_slice_ptr, (num_slices + 1) * sizeof(unsigned int));
    cudaMalloc(&d_col_ind, total_nnz_padded * sizeof(unsigned int));
    cudaMalloc(&d_vals, total_nnz_padded * sizeof(double));
    cudaMalloc(&d_x, m * sizeof(double));
    cudaMalloc(&d_y, m * sizeof(double));

    // 4. Copy Data
    cudaMemcpy(d_slice_ptr, h_sell_slice_ptr, (num_slices + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, h_sell_col_ind, total_nnz_padded * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_sell_vals, total_nnz_padded * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, m * sizeof(double), cudaMemcpyHostToDevice);

    // 5. Run Kernel
    spmv_gpu_sellc(m, SLICE_THICKNESS, num_slices, d_slice_ptr, d_col_ind, d_vals, d_x, d_y);

    // 6. Check Result
    double* h_y = (double*)malloc(m * sizeof(double));
    cudaMemcpy(h_y, d_y, m * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Result Y: ");
    for(int i=0; i<m; i++) printf("%f ", h_y[i]);
    printf("\n");

    return 0;
}