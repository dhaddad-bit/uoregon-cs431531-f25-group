#include "csr5.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Helper: to check for CUDA errors (My saving grace for debugging)
#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

// Helper: unpack the popcount of a 32-bit integer (our packed [bitflag|y_offset|seg_offset])
__device__ inline void unpack_tile_desc(unsigned int desc, unsigned int &bitflag, int &y_offset,
                                        int &seg_offset
) {
    seg_offset = desc & 0x7F;
    y_offset = (desc >> 7) & 0x1FF;
    bitflag = (desc >> 16);
}

__global__ void spmv_csr5_kernel(
    int num_tiles,
    const double* __restrict__ val,
    const int* __restrict__ col_idx,
    const int* __restrict__ tile_ptr,
    const unsigned int* __restrict__ tile_desc,
    const double* __restrict__ x,
    double* __restrict__ y
) {
    int tid = blockIdx.x;
    int lane_id = threadIdx.x; // OMEGA (0, ... ,31)

    if (tid >= num_tiles) return;
    
    // --- STEP 1: Fetch Descriptors --- 
    unsigned int desc = tile_desc[tid * CSR5_OMEGA + lane_id];
    unsigned int bit_flag;
    int y_offset, seg_offset;
    unpack_tile_desc(desc, bit_flag, y_offset, seg_offset);

    // --- STEP 2: Determine location in global row space --- 
    int tile_row_start = tile_ptr[tid];

    // --- STEP 3: Local summation (Vertical Reduction) --- 
    int tile_base = tid * CSR5_OMEGA * CSR5_SIGMA;

    // Iterate ver sigma 0..15
    int current_row_offset = y_offset;

    for (int j=0; j < CSR5_SIGMA; j++) {
        int idx = tile_base + lane_id * CSR5_SIGMA + j;
        // Read this tranposed data (Coalesced access!) NOTE:
        double v = val[idx];
        int c = col_idx[idx];
        // Check bit flag to see if this is a new row (or even valid)
        bool is_new_row = (bit_flag >> (CSR5_SIGMA -1 -j)) & 1;
        if (is_new_row) {
            current_row_offset++;
        }
        // Calculate Global Row Index
        int global_row = tile_row_start + current_row_offset;
        // TODO !!!!!!!!!! SIMPLIFICATION !!!!!!!!!!
        // ATOMIC ADD instead of 1000 lines of scanning and stuff I DO NOT UNDERSTAND!
        if (v != 0.0) { // the least we can do is skip zeros (STRESS)
            atomicAdd(&y[global_row], v * x[c]);
        }
    }
}

extern "C" void spmv_gpu_csr5(
    int m,
    int num_tiles,
    double* d_val,
    int* d_col_idx,
    int* d_tile_ptr,
    unsigned int* d_tile_desc,
    double* d_x,
    double* d_y
) {
    dim3 block(CSR5_OMEGA);
    dim3 grid(num_tiles);
    // Reset the output vector since we do atomic adds
    // Reason: we may have multiple threads writing to the same output
    cudaMemset(d_y, 0, m *sizeof(double));
    
    spmv_csr5_kernel<<<grid, block>>>(
        num_tiles, d_val, d_col_idx, d_tile_ptr, d_tile_desc, d_x, d_y
    );
    checkCuda(cudaGetLastError());
}
                                            
                                            