/**
 * Timing on GPU is different han on CPU
 * because of the asynchronous nature of GPU operations.
 * This file provides timing utilities specific to GPU.
 * 
 */
// This is the first time I've written code like this, 
// so Gemini did help walk me through the process and explain the new concepts
// but everything written here is my own work
// I'm doing my best to understand each line of code and why/where it is needed.

#include <cuda_runtime.h>
#include "spmv.h"
#include <iostream>
#include <cassert>

float launch_and_time_sell_c(
    int m_padded,
    int n,
    int num_slices,
    int total_nnz,
    int SLICE_THICKNESS,
    const unsigned int* h_sell_slice_ptr,
    const unsigned int* h_sell_col_ind,
    const P_TYPE* h_sell_vals,
    const P_TYPE* h_x;
    const P_TYPE* h_y; 
    )
{
    // --- 1. Device pointers ---
    unsigned int* d_sell_slice_ptr;
    unsigned int* d_sell_col_ind;
    P_TYPE* d_sell_vals;
    P_TYPE* d_x;
    P_TYPE* d_y;

    // --- 2. CUDA Events for Timing events ---
    cudaEvent_t start, stop;
    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    // --- 3. Allocate memory on device AKA GPU --- 
    gpuErrchk( cudaMalloc(&d_sell_slice_ptr, (num_slices+1)*sizeof(unsigned int)) );
    gpuErrchk( cudaMalloc(&d_sell_col_ind, total_nnz*sizeof(unsigned int)) );
    gpuErrchk( cudaMalloc(&d_sell_vals, total_nnz*sizeof(P_TYPE)) );
    gpuErrchk( cudaMalloc(&d_x, n*sizeof(P_TYPE)) );
    gpuErrchk( cudaMalloc(&d_y, m_padded*sizeof(P_TYPE)) );

    // --- 4. Host device memory copy ---
    gpuErrchk( cudaMemcpy(d_sell_slice_ptr, h_sell_slice_ptr, (num_slices + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_sell_col_ind, h_sell_col_ind, total_nnz * sizeof(unsigned int), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_sell_vals, h_sell_vals, total_nnz * sizeof(P_TYPE), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_x, h_x, n * sizeof(P_TYPE), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_y, h_y, m_padded * sizeof(P_TYPE), cudaMemcpyHostToDevice) );

    // --- 5. Set Up kernel launch parameters ---
    int THREADS_PER_BLOCK = 256;
    int GRID_SIZE = (m_padded + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // --- 6. Launch and time the kernel ---
    gpuErrchk( cudaEventRecord(start) );
    spmv_sell_c_kernel<<<GRID_SIZE, THREADS_PER_BLOCK>>>(
        m_padded,
        SLICE_THICKNESS,
        d_sell_slice_ptr,
        d_sell_col_ind,
        d_sell_vals,
        d_x,
        d_y
    );
    gpuErrchk( cudaEventRecord(stop) );

    // --- 7. Synchronize + calculate elapsed time ---
    gpuErrchk( cudaEventSynchronize(stop) );
    float milliseconds = 0;
    gpuErrchk( cudaEventElapsedTime(&milliseconds, start, stop) );

    // --- 8. Copy Result Vector back to host from device ---
    gpuErrchk( cudaMemcpy(h_y, d_y, m_padded * sizeof(P_TYPE), cudaMemcpyDeviceToHost) );

    // --- 9. Cleanup: free in opposite order of allocation (inside first for data structures) --- 
    gpuErrchk( cudaFree(d_y) );
    gpuErrchk( cudaFree(d_x) );
    gpuErrchk( cudaFree(d_sell_vals) );
    gpuErrchk( cudaFree(d_sell_col_ind) );
    gpuErrchk( cudaFree(d_sell_slice_ptr) );
    gpuErrchk( cudaEventDestroy(start) );
    gpuErrchk( cudaEventDestroy(stop) );

    // --- 10. Return elapsed time ---
    return milliseconds;

}
