#include <cuda_runtime.h>
#include <stdio.h>

// Simple error checking wrapper for cuda calls (youtube video)
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", 
                file, line, (unsigned int)result, cudaGetErrorString(result), func);
            exit(EXIT_FAILURE);
    }
}

__global__ void spmv_sell_c_kernel(
    int m,
    int num_slices,
    int SLICE_THICKNESS,
    const unsigned int* slice_ptr,
    const unsigned int* col_ind,
    const double* vals,
    const double* x,
    double* y
) {
    // 1. Identify the slice we are handling
    int slice_id = blockIdx.x;
    if (slice_id >= num_slices) return; // Boundary check

    // 2. Which row within the slice is this thread handling?
    int i = threadIdx.x;
    int row = slice_id * SLICE_THICKNESS + i;
    if (row >= m) return; // Boundary check for padded rows // IS THIS NECESSARY? TODO
    
    // 3. Determine the boundaries of the slice from slice pointer array
    unsigned int start = slice_ptr[slice_id];
    unsigned int next_start = slice_ptr[slice_id + 1];
    unsigned int slice_len = (next_start - start) / SLICE_THICKNESS;

    double sum = 0.0;

    // 4. Iterate over columns in slice (j = depth into slice)
    for (int j = 0; j < slice_len; j++) {
        // 5. Use/Match indexing from conversion_csr_sellc.cc:
        // index = start + (j*SLICE_THICKNESS) + i
        unsigned int idx = start + (j * SLICE_THICKNESS) + i;
        unsigned int col = col_ind[idx];
        double val = vals[idx];

        sum += val * x[col];
    }
    y[row] = sum;
}

// Wrapper function to call from main.cc (C-style linking needed for errors)
extern "C" void spmv_gpu_sellc(
    int m,
    int num_slices,
    int SLICE_THICKNESS,
    unsigned int* d_slice_ptr,
    unsigned int* d_col_ind,
    double* d_vals,
    double* d_x,
    double* d_y
) {
    // Launch a block per slice, with block size == slice thickness
    dim3 block(SLICE_THICKNESS);
    dim3 grid(num_slices);
    spmv_sell_c_kernel<<<grid, block>>>(m, num_slices, SLICE_THICKNESS, d_slice_ptr, d_col_ind, d_vals, d_x, d_y);
    // Check for kernel launch errors
    checkCudaErrors(cudaGetLastError());
}



