#include <iostream>
#include <stdio.h>
#include <assert.h>

//#include <helper_cuda.h>
#include <cooperative_groups.h>

#include "gpu_spmv.h"

//Not finished yet!!!


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
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = threadIdx.x % 32;

    int row = warp_id; // just 1 warp per row

    if (row < m) {
        double prod = 0;
        for (int idx = row_ptr[row] + lane_id; idx < row_ptr[row + 1]; idx += 32) {
            prod += vals[idx] * x[col_ind[idx]];
        }

        // warp level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            prod += __shfl_down_sync(0xffffffff, prod, offset);
        }
    
        if (lane_id == 0) {
            b[row] += prod;
        }
    }
}

__host__ void vector_csr(unsigned int* col_ind, unsigned int* row_ptr, double* vals, int m, int n, int nnz, 
                    double* x, double* b) 
{
    int threads_per_block = 256;
    dim3 block(threads_per_block);
    int blocks_per_grid = (m + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_per_grid);

    vector_csr_kernel<<<grid, block>>>(col_ind, row_ptr, vals, m, n, nnz, x, b);
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
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = threadIdx.x % 32;
    int ind;

    for (ind = tid; ind < nnz; ind += 32) {
        double val = vals[ind] * x[col_ind[ind]];
        // segmented scan according to current row
        int prev_row = __shfl_up_sync(0xffffffff, row_ind[ind], 1);
        bool new_row = false;
        if (lane_id == 0 || row_ind[ind] != prev_row) {
            new_row = true;
        }
        for (int offset = 1; offset < 32; offset *= 2) {
            double tmp = __shfl_up_sync(0xffffffff, val, offset);
            int tmp_row = __shfl_up_sync(0xffffffff, row_ind[ind], offset);
            if (lane_id >= offset && tmp_row == row_ind[ind]) {
                val += tmp;
            }
        }
        if (new_row == true) {
            atomicAdd(&b[row_ind[ind]], val);
        }
    }
}

__host__ void vector_coo(unsigned int *col_ind, unsigned int *row_ind, double *vals, int m, int n, int nnz,
                        double *x, double *b)
{
    int threads_per_block = 256;
    dim3 block(threads_per_block);
    int blocks_per_grid = (m + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_per_grid);

    vector_coo_kernel<<<grid, block>>>(col_ind, row_ind, vals, m, n, nnz, x, b);
}