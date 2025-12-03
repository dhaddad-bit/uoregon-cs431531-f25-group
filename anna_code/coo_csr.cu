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
    for (int i = 0; i < nnz; i++) {
        b[row_ind[i]] += vals[i] * x[col_ind[i]];
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

__global__ void vector_coo_kernel(unsigned int *col_ind, unsigned int *row_ind, double *vals, int m, int n, int nnz,
                        double *x, double *b)
{
    // get thread id and position in warp
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // initial access will be tid
    int ind = tid;
    int row;
    double val;

    // make sure not outside of range
    if (ind < nnz) {
        row = row_ind[ind];
        val = vals[ind] * x[col_ind[ind]];
    } else {
        row = -1;
        val = 0.0;
    }

    // using a while loop to follow more closely to pseudocode
    while (ind < nnz) {

        int next_ind = ind + WARP_SIZE;
        int next_row = (next_ind < nnz ? row_ind[next_ind] : row);

        // segment will end if any lane sees different next_row
        bool bound_next_iter = __any_sync(0xffffffff, next_row != row);

        // need to check within warp
        int row_down = __shfl_down_sync(0xffffffff, row, 1);
        bool bound_warp = (row != row_down);

        // if change rows in this or next iteration, need to flush sums
        if ((bound_warp || bound_next_iter) && row >= 0) {
            // segmented scan
            double acc = val;
            for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
                double tmp_val = __shfl_up_sync(0xffffffff, acc, offset);
                int tmp_row = __shfl_up_sync(0xffffffff, row, offset);
                if (lane_id >= offset && tmp_row == row) {
                    acc += tmp_val;
                }
            }

            // segment leader writes result
            int row_up = __shfl_up_sync(0xffffffff, row, 1);
            if ((lane_id == 0 || row != row_up) && row >= 0) {
                atomicAdd(&b[row], acc);
            }
        }

        ind = next_ind;
        row = next_row;

        if (ind < nnz) {
            val = vals[ind] * x[col_ind[ind]];
        } else {
            val = 0.0;
        }
    }
}

__host__ void vector_coo(unsigned int *col_ind, unsigned int *row_ind, double *vals, int m, int n, int nnz,
                        double *x, double *b)
{
    int threads_per_block = 256;
    int blocks_per_grid = (nnz + threads_per_block - 1) / threads_per_block; 

    vector_coo_kernel<<<blocks_per_grid, threads_per_block>>>(col_ind, row_ind, vals, m, n, nnz, x, b);
}