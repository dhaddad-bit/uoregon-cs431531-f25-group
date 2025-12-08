#include <iostream>
#include <stdio.h>
#include <assert.h>

//#include <helper_cuda.h>
#include <cooperative_groups.h>

#include "gpu_spmv.h"
#include "spmv.h"


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
inline void check_cuda(cudaError_t result, const char* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

#define WARP_SIZE 32

void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
                      P_TYPE* vals, int m, int n, int nnz, P_TYPE* x, 
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      P_TYPE** dev_vals, P_TYPE** dev_x, P_TYPE** dev_b)
{
    // allocate memory for csr data
    unsigned int row_size = sizeof(unsigned int) * (m + 1);
    unsigned int col_size = sizeof(unsigned int) * nnz;
    unsigned int val_size = sizeof(P_TYPE) * nnz;
    unsigned int x_size = sizeof(P_TYPE) * n;

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
    cudaMalloc(dev_b, sizeof(P_TYPE) * m);
}

void allocate_coo_gpu(unsigned int* row_ind, unsigned int* col_ind, 
                      P_TYPE* vals, int m, int n, int nnz, P_TYPE* x, 
                      unsigned int** dev_row_ind, unsigned int** dev_col_ind,
                      P_TYPE** dev_vals, P_TYPE** dev_x, P_TYPE** dev_b)
{
    // allocate memory for csr data
    unsigned int row_size = sizeof(unsigned int) * nnz;
    unsigned int col_size = sizeof(unsigned int) * nnz;
    unsigned int val_size = sizeof(P_TYPE) * nnz;
    unsigned int x_size = sizeof(P_TYPE) * n;

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
    cudaMalloc(dev_b, sizeof(P_TYPE) * m);
    cudaMemset(*dev_b, 0, m * sizeof(P_TYPE));
}

void get_result_gpu(P_TYPE* dev_b, P_TYPE* b, int m)
{
    cudaMemcpy(b, dev_b, sizeof(P_TYPE) * m, cudaMemcpyDeviceToHost);
    cudaFree(dev_b);
}

__global__ void scalar_csr_kernel(unsigned int* col_ind, unsigned int* row_ptr, P_TYPE* vals, int m, int n, int nnz, 
                        P_TYPE* x, P_TYPE* b) 
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        P_TYPE prod = 0;
        for (int i = row_ptr[row]; i < row_ptr[row+1]; i++) {
            prod += vals[i] * x[col_ind[i]];
        }
        b[row] = prod;
    }
}

__host__ void scalar_csr(unsigned int* col_ind, unsigned int* row_ptr, P_TYPE* vals, int m, int n, int nnz, 
                    P_TYPE* x, P_TYPE* b, float* time_ms) 
{
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    int threads_per_block = 256;
    dim3 block(threads_per_block);
    int blocks_per_grid = (m + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_per_grid);

    checkCudaErrors(cudaEventRecord(start, 0));

    //for (int i = 0; i < MAX_ITER; i++) {
        scalar_csr_kernel<<<grid, block>>>(col_ind, row_ptr, vals, m, n, nnz, x, b);
    //}

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    *time_ms = elapsedTime; // / (float)MAX_ITER;

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
}

__global__ void vector_csr_kernel(unsigned int* col_ind, unsigned int* row_ptr, P_TYPE* vals, int m, int n, int nnz, 
                    P_TYPE* x, P_TYPE* b) 
{
    //
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    
    int row_idx = blockIdx.x * warps_per_block + warp; // just 1 warp per row
    int subrow = gridDim.x * warps_per_block;

    P_TYPE prod = 0.0;

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

__host__ void vector_csr(unsigned int* col_ind, unsigned int* row_ptr, P_TYPE* vals, int m, int n, int nnz, 
                    P_TYPE* x, P_TYPE* b, float* time_ms) 
{
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks_per_grid = (m + warps_per_block - 1) / warps_per_block;

    checkCudaErrors(cudaEventRecord(start, 0));

    //for (int i = 0; i < MAX_ITER; i++) {
        vector_csr_kernel<<<blocks_per_grid, threads_per_block>>>(col_ind, row_ptr, vals, m, n, nnz, x, b);
    //}
    
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    *time_ms = elapsedTime; // / (float)MAX_ITER;

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
}

__global__ void scalar_coo_kernel(unsigned int* col_ind, unsigned int* row_ind, P_TYPE* vals, int m, int n, int nnz, 
                    P_TYPE* x, P_TYPE* b) 
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < nnz) {
        int row = row_ind[ind];
        int col = col_ind[ind];
        P_TYPE val = vals[ind];

        atomicAdd(&b[row], val * x[col]);
    }
}

__host__ void scalar_coo(unsigned int* col_ind, unsigned int* row_ind, P_TYPE* vals, int m, int n, int nnz, 
                    P_TYPE* x, P_TYPE* b, float* time_ms)
{
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    int threads_per_block = 256;
    dim3 block(threads_per_block);
    int blocks_per_grid = (m + threads_per_block - 1) / threads_per_block;
    dim3 grid(blocks_per_grid);
    
    checkCudaErrors(cudaEventRecord(start, 0));
    //for (int i = 0; i < MAX_ITER; i++) {
        scalar_coo_kernel<<<grid, block>>>(col_ind, row_ind, vals, m, n, nnz, x, b);
    //}
    
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    *time_ms = elapsedTime; /// (float)MAX_ITER;

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
}
