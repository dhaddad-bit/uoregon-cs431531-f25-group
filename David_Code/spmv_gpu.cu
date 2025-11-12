/**
 * Kernel for SpMV using SELL-C format.
 * 
 * 
 */

 __global__ void spmv_sell_c_kernel(
    int m_padded,
    int SLICE_THICKNESS,
    const unsigned int* sell_slice_ptr,
    const unsigned int* sell_col_ind,
    const double* sell_vals,
    const double* x,
    double* y
    )
 {
    // 1. Get the global row ID for this thread
    int global_row = blockIdx.x * blockDIm.x + threadIdx.x;
    // 2. Bound check, no work for a thread if its outside of padded range for matrix
    if (global_row >= m_padded) {
        return;
    }
    double sum = 0.0;
    // 3. Find which slice each row belongs to
    int slice_id = global_row / SLICE_THICKNESS;
    // 4. Find this thread's position within the slice (0 to SLICE_THICKNESS-1) range
    int thread = global_row % SLICE_THICKNESS;
    // 5. Get slice boundaries and length
    unsigned int slice_s = sell_slice_ptr[slice_id];
    unsigned int slice_e = sell_slice_ptr[slice_id + 1];
    unsigned int slice_length = (slice_e - slice_s) / SLICE_THICKNESS;
    // 6. Iterate over columns in this slice
    // Workload for each thread in the slice should be the same
    for (int j=0; j<slice_length; j++) {
        // 7. Compute column-major order for coalescing (GEMINI HELP ME UNDERSTAND THIS)
        int idx = slice_start + j * SLICE_THICKNESS + thread;
        // 8. Get col index and value
        unsigned int col = sell_col_ind[idx];
        double val = sell_vals[idx];
        sum += val * x[col];
    }
    y[global_row] = sum;
 }