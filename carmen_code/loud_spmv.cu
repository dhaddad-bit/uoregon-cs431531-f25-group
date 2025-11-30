#include <iostream>
#include <stdio.h>
#include <assert.h>

// Simple error checking wrapper for cuda calls (youtube video) compiler can't find it for some reason
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
inline void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", 
                file, line, (unsigned int)result, cudaGetErrorString(result), func);
            exit(EXIT_FAILURE);
    }
}


#include <cooperative_groups.h> 
#include "spmv.h"

#define MAX_ITER 100
template <class T>
__global__ void
spmv_kernel_ell(unsigned int* col_ind, T* vals, int m, int n, int nnz, 
                double* x, double* b)
{
    // COMPLETE THIS FUNCTION
    // use one BLOCK per row
    // n = max_nonzero_per_row (padded)
    // nnz = original number of non_zeros (not padded value)
    extern __shared__ double data[];
    unsigned int row = blockIdx.x; // new: BLOCK = ROW we are processing
    unsigned int tid = threadIdx.x; // Thread ID inside the block (0 to 63)! Remember this indexing!

    double partial_sum = 0.0;
    
    // --- calculate row indexes in ELL format --- 
    unsigned int row_start = row * n;
    unsigned int row_end = row_start+n; // Constant in ell as apposed to csr
    // Perfectly coallesed access within the block (ideally)
    for (unsigned int j = row_start+tid; j < row_end; j+= blockDim.x) { // note: j increments by blockDim.x
        // be careful to not go out of bounds of original nnz
        unsigned int col = col_ind[j];
        if (col != (unsigned int)-1) {
            partial_sum += vals[j] * x[col];
        }
    }

    // --- store the partial sums in shared memory (data) --- 
    data[tid] = partial_sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s>0; s>>=1) {
        if (tid < s) {
            data[tid] += data[tid +s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        b[row] = data[0];
    }
}



void spmv_gpu_ell(unsigned int* col_ind, double* vals, int m, int n, int nnz, 
                  double* x, double* b, unsigned int threads, float* time_ms)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // GPU execution parameters
    unsigned int blocks = m; 
    // unsigned int threads = 64; 
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel_ell<double><<<dimGrid, dimBlock, shared>>>(col_ind, vals, 
                                                               m, n, nnz, x, b);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    *time_ms = elapsedTime / (float)MAX_ITER;
    // printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
}




void allocate_ell_gpu(unsigned int* col_ind, double* vals, int m, int n, 
                      int nnz, double* x, unsigned int** dev_col_ind, 
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // copy ELL data to GPU and allocate memory for output
    // COMPLETE THIS FUNCTION
    // n = "n_new" (max_nnz_per_row) from CPU main.cc function
    // nnz = nnz remains the same
    // but! padded value for allocation of memory on GPU ...
    int padded_nnz = m*n;

    // Use 'CopyData' as described to allocate memory for ELL arrays on device (GPU)
    CopyData(col_ind, padded_nnz, sizeof(unsigned int), dev_col_ind);
    CopyData(vals, padded_nnz, sizeof(double), dev_vals);

}

void allocate_csr_gpu(unsigned int* row_ptr, unsigned int* col_ind, 
                      double* vals, int m, int n, int nnz, double* x, 
                      unsigned int** dev_row_ptr, unsigned int** dev_col_ind,
                      double** dev_vals, double** dev_x, double** dev_b)
{
    // copy CSR data to GPU and allocate memory for output
    // COMPLETE THIS FUNCTION
    // Allocate memory for CCR arrays on device (GPU) using 'CopyData'
    CopyData(row_ptr, m+1, sizeof(unsigned int), dev_row_ptr);
    CopyData(col_ind, nnz, sizeof(unsigned int), dev_col_ind);
    CopyData(vals, nnz, sizeof(double), dev_vals);

    // Allocate device (GPU) memory for vectors using 'CopyData'
    CopyData(x, n, sizeof(double), dev_x);
    
    // allocate (not copy) b on device (GPU)
    checkCudaErrors(cudaMalloc((void**) dev_b, sizeof(double) * m));
}

void get_result_gpu(double* dev_b, double* b, int m)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsedTime;


    checkCudaErrors(cudaEventRecord(start, 0));
    checkCudaErrors(cudaMemcpy(b, dev_b, sizeof(double) * m, 
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("  Pinned Host to Device bandwidth (GB/s): %f\n",
         (m * sizeof(double)) * 1e-6 / elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template <class T>
void CopyData(
  T* input,
  unsigned int N,
  unsigned int dsize,
  T** d_in)
{
  // timers
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;

  // Allocate pinned memory on host (for faster HtoD copy)
  T* h_in_pinned = NULL;
  checkCudaErrors(cudaMallocHost((void**) &h_in_pinned, N * dsize));
  assert(h_in_pinned);
  memcpy(h_in_pinned, input, N * dsize);

  // copy data
  checkCudaErrors(cudaMalloc((void**) d_in, N * dsize));
  checkCudaErrors(cudaEventRecord(start, 0));
  checkCudaErrors(cudaMemcpy(*d_in, h_in_pinned,
                             N * dsize, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("  Pinned Device to Host bandwidth (GB/s): %f\n",
         (N * dsize) * 1e-6 / elapsedTime);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}


template <class T>
__global__ void
spmv_kernel(unsigned int* row_ptr, unsigned int* col_ind, T* vals, 
            int m, int n, int nnz, double* x, double* b)
{
    // COMPLETE THIS FUNCTION
    extern __shared__ double data[];
    unsigned int row = blockIdx.x;  // new: BLOCK = ROW we are processing
    unsigned int tid = threadIdx.x; // Thread ID inside the block (0 to 63)! Remember this indexing!

    // --- partial sum for this thread --- 
    double partial_sum = 0.0;
    unsigned int row_start = row_ptr[row];
    unsigned int row_end = row_ptr[row + 1];
    // blocks traverse/coellese data in the row differently 
    for (unsigned int j = row_start + tid; j<row_end; j+= blockDim.x) {
        partial_sum += vals[j] * x[col_ind[j]];
    }

    // --- store partial sum in shared memory --- 
    data[tid] = partial_sum;
    __syncthreads();

    // parallel reduction in shared memory
    // (just starting with sequential reduction)
    for (unsigned int s = blockDim.x / 2; s>0; s>>=1) {
        if (tid < s) {
            data[tid] += data[tid +s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        b[row] = data[0];
    }
}


void spmv_gpu(unsigned int* row_ptr, unsigned int* col_ind, double* vals,
              int m, int n, int nnz, double* x, double* b, unsigned int threads, float* time_ms)
{
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float elapsedTime;

    // GPU execution parameters
    // 1 thread block per row
    // 64 threads working on the non-zeros on the same row
    unsigned int blocks = m; 
    // unsigned int threads = 64; 
    unsigned int shared = threads * sizeof(double);

    dim3 dimGrid(blocks, 1, 1);
    dim3 dimBlock(threads, 1, 1);

    checkCudaErrors(cudaEventRecord(start, 0));
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_kernel<double><<<dimGrid, dimBlock, shared>>>(row_ptr, col_ind, 
                                                           vals, m, n, nnz, 
                                                           x, b);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    *time_ms = elapsedTime / (float)MAX_ITER;
    // printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));

}


// --- SELL-C-sigma SpMV kernel --- 
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
        if (col != (unsigned int)(-1)) {
            // Valid entry
            double val = vals[idx];
            sum += val * x[col];
        }
    }
    if (row < m) {
        y[row] = sum;
    }
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
    double* d_y,
    float* time_ms
) {
    // timers
    cudaEvent_t start;
    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // GPU execution parameters
    dim3 block(SLICE_THICKNESS);
    dim3 grid(num_slices);
    // Record start:
    checkCudaErrors(cudaEventRecord(start, 0));
    // Run max iterations
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        cudaDeviceSynchronize();
        spmv_sell_c_kernel<<<grid, block>>>(
            m,
            num_slices,
            SLICE_THICKNESS,
            d_slice_ptr,
            d_col_ind,
            d_vals,
            d_x,
            d_y
        );
    }
    // Record Stop
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    *time_ms = elapsedTime / (float)MAX_ITER;
    // printf("  Exec time (per itr): %0.8f s\n", (elapsedTime / 1e3 / MAX_ITER));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaGetLastError());
}  
