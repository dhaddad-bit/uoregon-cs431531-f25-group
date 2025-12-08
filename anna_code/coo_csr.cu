#include <iostream>
#include <stdio.h>
#include <assert.h>

//#include <helper_cuda.h>
#include <cooperative_groups.h>

#include "gpu_spmv.h"

#define WARP_SIZE 32

void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
                      double* vals, int m, int n, int nnz, double* x, 
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // allocate memory for csr data
    unsigned int row_size = sizeof(unsigned int) * (m + 1);
    unsigned int col_size = sizeof(unsigned int) * nnz;
    unsigned int val_size = sizeof(double) * nnz;
    unsigned int x_size = sizeof(double) * n;

    cudaMalloc(dev_row_ptr, row_size);
    cudaMalloc(dev_col_ind, col_size);
    cudaMalloc(dev_vals, val_size);
    cudaMalloc(dev_x, x_size);

    // copy data from CPU to GPU
    cudaMemcpy(*dev_row_ptr, row_ptr, row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_col_ind, col_ind, col_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_vals, vals, val_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_x, x, x_size, cudaMemcpyHostToDevice);

    // allocate memory for results (b)
    cudaMalloc(dev_b, sizeof(double) * m);
}

void allocate_coo_gpu(unsigned int* row_ind, unsigned int* col_ind, 
                      double* vals, int m, int n, int nnz, double* x, 
                      unsigned int** dev_row_ind, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // allocate memory for csr data
    unsigned int row_size = sizeof(unsigned int) * nnz;
    unsigned int col_size = sizeof(unsigned int) * nnz;
    unsigned int val_size = sizeof(double) * nnz;
    unsigned int x_size = sizeof(double) * n;

    cudaMalloc(dev_row_ind, row_size);
    cudaMalloc(dev_col_ind, col_size);
    cudaMalloc(dev_vals, val_size);
    cudaMalloc(dev_x, x_size);

    // copy data from CPU to GPU
    cudaMemcpy(*dev_row_ind, row_ind, row_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_col_ind, col_ind, col_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_vals, vals, val_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*dev_x, x, x_size, cudaMemcpyHostToDevice);

    // allocate memory for results (b)
    cudaMalloc(dev_b, sizeof(double) * m);
    cudaMemset(*dev_b, 0, m * sizeof(double));
}

void get_result_gpu(double* dev_b, double* b, int m)
{
    cudaMemcpy(b, dev_b, sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaFree(dev_b);
}

__global__ void scalar_csr_kernel(unsigned int* col_ind, unsigned int* row_ptr, double* vals, int m, int n, int nnz, 
                        double* x, double* b) 
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        double prod = 0;
        for (int i = row_ptr[row]; i < row_ptr[row+1]; i++) {
            prod += vals[i] * x[col_ind[i]];
        }
        b[row] = prod;
    }
}

__host__ void scalar_csr(unsigned int* col_ind, unsigned int* row_ptr, double* vals, int m, int n, int nnz, 
                    double* x, double* b) 
{
    int threads_per_block = 256;
    dim3 block(threads_per_block);
    int blocks_per_grid = (m + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_per_grid);

    scalar_csr_kernel<<<grid, block>>>(col_ind, row_ptr, vals, m, n, nnz, x, b);
}

__global__ void vector_csr_kernel(unsigned int* col_ind, unsigned int* row_ptr, double* vals, int m, int n, int nnz, 
                    double* x, double* b) 
{
    //
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    
    int row_idx = blockIdx.x * warps_per_block + warp; // just 1 warp per row
    int subrow = gridDim.x * warps_per_block;

    double prod = 0.0;

    for (int row = row_idx; row < m; row += subrow) {
        prod = 0.0;
        for (int idx = row_ptr[row] + lane; idx < row_ptr[row + 1]; idx += 32) {
            prod += vals[idx] * x[col_ind[idx]];
        }

        // warp level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            prod += __shfl_down_sync(0xffffffff, prod, offset);
        }
    
        if (lane == 0) {
            b[row] += prod;
        }
    }
}

__host__ void vector_csr(unsigned int* col_ind, unsigned int* row_ptr, double* vals, int m, int n, int nnz, 
                    double* x, double* b) 
{
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks_per_grid = (m + warps_per_block - 1) / warps_per_block;
    
    vector_csr_kernel<<<blocks_per_grid, threads_per_block>>>(col_ind, row_ptr, vals, m, n, nnz, x, b);
}

__global__ void scalar_coo_kernel(unsigned int* col_ind, unsigned int* row_ind, double* vals, int m, int n, int nnz, 
                    double* x, double* b) 
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < nnz) {
        int row = row_ind[ind];
        int col = col_ind[ind];
        double val = vals[ind];

        atomicAdd(&b[row], val * x[col]);
    }
}

__host__ void scalar_coo(unsigned int* col_ind, unsigned int* row_ind, double* vals, int m, int n, int nnz, 
                    double* x, double* b)
{
    int threads_per_block = 256;
    dim3 block(threads_per_block);
    int blocks_per_grid = (m + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_per_grid);
    scalar_coo_kernel<<<grid, block>>>(col_ind, row_ind, vals, m, n, nnz, x, b);
}

/*
Algorithm 1. Load-balancing COO kernel algorithm.
1: Get ind = index of the first element to be processed by this thread
2: Get current row = rowidx[ind].
3: Compute the first value c = A[ind] × x[colidx[ind]]
4: for i = 0 .. nz per warp; i+ = warpsize do
5:      Compute next row, row index of the next element to be processed
6:      if any thread in the warp’s next row != current row or it is the final iteration
    then
7:          Compute the segmented scan according to current row.
8:          if first thread in segment then
9:              atomicAdd c on output vector by the first entry of each segment
10:         end if
11:         Reinitialize c = 0
12:     end if
13:     Get the next index ind
14:     Compute c+ = A[ind] × x[colidx[ind]]
15:     Update current row to next row
16: end for
*/

__global__ void vector_coo_kernel(unsigned int *col_ind, unsigned int *row_ind, double *vals, int m, int n, int nnz,
                        double *x, double *b)
{
    // get thread id and position in warp
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // make sure not outside of range
    int start = warp_id * WARP_SIZE;
    int ind = start + lane_id;

    // load initial data
    int row = (ind < nnz) ? row_ind[ind] : -1;
    double val = (ind < nnz) ? vals[ind] * x[col_ind[ind]] : 0.0;

    // process all non-zeros
    while (true) {
        unsigned int active = __activemask();
        unsigned int valid_mask = __ballot_sync(active, ind < nnz);

        // segmented scan
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset <<= 2) {
            // get value and row of thread offset above
            double tmp_val = __shfl_up_sync(valid_mask, val, offset);
            int tmp_row = __shfl_up_sync(valid_mask, row, offset);

            // if same row, add value
            if (lane_id >= offset && tmp_row == row) {
                val += tmp_val;
            }
        }

        int row_down = __shfl_down_sync(valid_mask, row, 1);

        // if last in segment, write result (rather than first in segment)
        if (lane_id == WARP_SIZE - 1 || row != row_down) {
            if (row >= 0 && row < m) {
                atomicAdd(&b[row], val);
            }
        }
    
        // get next index
        ind = ind + WARP_SIZE;
        if (ind >= nnz) {
            break;
        }

        // load data for next iteration
        row = row_ind[ind];
        val = vals[ind] * x[col_ind[ind]];
    }
}

__host__ void vector_coo(unsigned int *col_ind, unsigned int *row_ind, double *vals, int m, int n, int nnz,
                        double *x, double *b)
{
    int threads = 256;
    // each warp processes WARP_SIZE non-zeros
    int num_warps = (nnz + WARP_SIZE - 1) / WARP_SIZE;

    // number of blocks needed based on number of warps and threads per block
    int blocks = (num_warps + (threads / WARP_SIZE) - 1) 
                     / (threads / WARP_SIZE);

    vector_coo_kernel<<<blocks, threads>>>(
        col_ind, row_ind, vals, m, n, nnz, x, b);
}

__global__ void simpler_vector_coo_kernel(unsigned int *col_ind, unsigned int *row_ind, double *vals, int m, int n, int nnz,
                        double *x, double *b)
{
    // get thread id and position in warp
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x % WARP_SIZE;

    unsigned int mask = __activemask(); // all threads active

    // iterate over all non-zeros
    for (int i = tid; i < nnz; i += gridDim.x * blockDim.x) {
        // load data
        int row = row_ind[i];
        double val = vals[i] * x[col_ind[i]];

        // warp level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            // get value and row of thread offset above
            int row_up = __shfl_up_sync(mask, row, offset);
            double val_up = __shfl_up_sync(mask, val, offset);
            // if same row, add value
            if (lane_id >= offset && row_up == row) {
                val += val_up;
            }
        }

        // last thread in segment writes result
        int row_down = __shfl_down_sync(mask, row, 1);
        bool is_last_in_segment = (lane_id == WARP_SIZE - 1) || (row != row_down);

        if (is_last_in_segment) {
            atomicAdd(&b[row], val);
        }
    }
}

__host__ void simpler_vector_coo(unsigned int *col_ind, unsigned int *row_ind, double *vals, int m, int n, int nnz,
                        double *x, double *b)
{
    // launch kernel
    int threads = 256;
    int blocks  = (nnz + threads - 1) / threads;

    simpler_vector_coo_kernel<<<blocks, threads>>>(
        col_ind, row_ind, vals, m, n, nnz, x, b);
}