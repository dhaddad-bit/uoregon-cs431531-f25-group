#include "csr5.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_ITER 100

// Helper: to check for CUDA errors (My saving grace for debugging)
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

// Helper macro for 2D access
#define MATRIX_ACCESS(ptr, pitch, row, col) \
    ((double*)((char*)(ptr) + (row) * (pitch)))[col]


__global__ void assign_csr5_val(double* og_val,
		int sigma, int omega,
		double* csr5_val, size_t pitch_val,
		int nnz, int num_tiles)
{
	//attempt to assign gpu values
	int tile_col = blockIdx.x; 
	int tile_row = blockIdx.y;
	int local_pos = threadIdx.x;
	//T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
	//
	int global_idx = (tile_row * num_tiles + tile_col) * sigma + local_pos;

	if (global_idx < nnz){
		int target_row = tile_row;
		int target_col = tile_col*sigma + local_pos;

		MATRIX_ACCESS(csr5_val, pitch_val, target_row, target_col) = og_val[global_idx];


        if (tile_row == 0 && tile_col == 0 && local_pos < 5) {
            printf("Kernel: Writing og_val[%d]=%.2f to csr5[%d][%d]\n", 
                   global_idx, og_val[global_idx], target_row, target_col);
        }
        
        if (target_col == 0 && target_row < 3) {
            printf("First column: csr5[%d][0] = %.2f\n", 
                   target_row, og_val[global_idx]);
        }
    }
    
    // Print after all threads complete their work (from thread 0 only)
    __syncthreads();
    if (tile_row == 0 && tile_col == 0 && local_pos == 0) {
        printf("Kernel completed: assigned %d values to %dx%d tiles\n", 
               nnz, omega, num_tiles);
    }
}


__global__ void assign_csr5_col(int* og_col,
                int sigma, int omega,
                int* csr5_col, size_t pitch_col,
                int nnz, int num_tiles)
{
	//attempt to assign gpu columns
	int tile_col = blockIdx.x; 
	int tile_row = blockIdx.y;
	int local_pos = threadIdx.x;
	//T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column;
	//
	int global_idx = (tile_row * num_tiles + tile_col) * sigma + local_pos;

	if (global_idx < nnz){
		int target_row = tile_row;
		int target_col = tile_col*sigma + local_pos;

		MATRIX_ACCESS(csr5_col, pitch_col, target_row, target_col) = og_col[global_idx];


        if (tile_row == 0 && tile_col == 0 && local_pos < 5) {
            printf("Kernel: Writing og_val[%d]=%d to csr5[%d][%d]\n", 
                   global_idx, og_col[global_idx], target_row, target_col);
        }
        
        if (target_col == 0 && target_row < 3) {
            printf("First column: csr5[%d][0] = %d\n", 
                   target_row, og_col[global_idx]);
        	}
        }
    
        __syncthreads();

        if (tile_row == 0 && tile_col == 0 && local_pos == 0) {
        	printf("Kernel completed: assigned %d values to %dx%d tiles\n", 
                nnz, omega, num_tiles);
    	}
}


/*
__global__ void gen_tile_ptr(int* tile_ptr,
		int sigma, int omega, 
		int* row_ptr){
	int tile_col = blockIdx.x;
	int tile_row = blockIdx.y;
	int local_pos = threadIdx.x;
	int global_tid = (tile_row * num_tiles +  tile_col) * sigma + local_pos;

	int bnd = local_pos * sigma * omega;
	//tile_ptr[local_pos] = binary_search(*row_ptr, bnd) -1;

		
something to come back to 
}*/


void convert_csr_to_csr5_gpu(
    //Inputs
    int m, int n, int nnz,
    const unsigned int* og_row_ptr,
    const unsigned int* og_col_idx,
    const double* og_val,
    int* sigma, int* omega,
    // Outputs
    int* num_tiles,
    double** gpu_csr5_val,
    int** gpu_csr5_col_idx,
    int** gpu_csr5_row_ptr, 
    int** gpu_csr5_tile_ptr,
    uint8_t** gpu_csr5_tile_desc
){	
	size_t pitch_val, pitch_col;
	//allocate memory for value blocks, 
	//cudaMallocPitch(gpu_csr5_val, omega, sigma * num_tiles);
	cudaMallocPitch((void**)gpu_csr5_val, &pitch_val, 
                   (*omega) * (*num_tiles) * sizeof(double),  // width in BYTES
                   *sigma);                                   // height in rows
        printf("Allocated with pitch: %zu bytes\n", pitch_val);
	//    T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column; 
	//   ^^ address of row, col of any element is given as above ^^
    	double* d_og_val;
    	cudaMalloc(&d_og_val, nnz * sizeof(double));
    	cudaMemcpy(d_og_val, og_val, nnz * sizeof(double), cudaMemcpyHostToDevice);
    	
	//ASSIGNING VALUES - LAUNCH KERNEL with <<< >>>
	dim3 blocks(*num_tiles, *omega);  // grid dimensions
	dim3 threads(*sigma);             // block dimensions
	    
	//ASSIGNING VALUES
	assign_csr5_val<<<blocks, threads>>>(
		d_og_val, 
		*sigma, *omega, 
		*gpu_csr5_val, pitch_val, 
		nnz, *num_tiles
	);	
	cudaDeviceSynchronize();  // waaait for kernel to finish  
	// then clean temporary GPU memory
	cudaFree(d_og_val);

    	int* d_og_col;
    	cudaMalloc(&d_og_col, nnz * sizeof(int));
    	cudaMemcpy(d_og_col, og_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
	
	// Similarly allocate other 2D arrays
    	cudaMallocPitch((void**)gpu_csr5_col_idx, &pitch_col, 
                   (*omega) * (*num_tiles) * sizeof(int), 
                   *sigma);
        printf("Allocated with pitch: %zu bytes\n", pitch_col);
	
	//ASSIGNING COLLUMNS
	assign_csr5_col<<<blocks, threads>>>(
			d_og_col, *sigma, 
			*omega, *gpu_csr5_col_idx, 
			pitch_col, nnz, *num_tiles
	);
	cudaDeviceSynchronize();  // waaait for kernel to finish  
	// then clean temporary GPU memory
	cudaFree(d_og_col);

	//1D allocation and can copy memory for rowptr since they the same
	cudaMalloc((void**)gpu_csr5_row_ptr, (m+1)*sizeof(int));
	cudaMemcpy(*gpu_csr5_row_ptr, og_row_ptr, (m+1)*sizeof(int), cudaMemcpyHostToDevice);
	
		//also mempry for the tile ptr
	//cudaMalloc((void**)gpu_csr5_tile_ptr,(*num_tiles)*sizeof(int));//maybe this should be unsigned int though
	
	//GENERATE ROW POINTER
	/*gen_tile_ptr<<<blocks, threads>>>(*gpu_csr5_tile_ptr,
		*sigma, *omega, 
		*gpu_csr5_row_ptr
	);*/

	

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
    const int* __restrict__ row_idx, // Row versioning
    const int* __restrict__ tile_ptr,
    const unsigned int* __restrict__ tile_desc,
    const double* __restrict__ x,
    double* __restrict__ y
) {
    int tid = blockIdx.x;
    int lane_id = threadIdx.x; // OMEGA (0, ... ,31)

    if (tid >= num_tiles) return;
    // --- ::Row versioning:: ---
    int tile_base = tid * CSR5_OMEGA * CSR5_SIGMA;
    // Coalesced Loop
    for (int j=0; j<CSR5_SIGMA; j++) {
        int idx = tile_base + lane_id * CSR5_SIGMA + j;
        int r = row_idx[idx];  // No packing!!!
        if (r != -1) {
            double v = val[idx];
            int c = col_idx[idx];
            atomicAdd(&y[r], v * x[c]);
        }
    }
    // --- ::Attempt1 Late night attempt:: ---
    // // --- STEP 1: Fetch Descriptors --- 
    // unsigned int desc = tile_desc[tid * CSR5_OMEGA + lane_id];
    // unsigned int bit_flag;
    // int y_offset, seg_offset;
    // unpack_tile_desc(desc, bit_flag, y_offset, seg_offset);

    // // --- STEP 2: Determine location in global row space --- 
    // int tile_row_start = tile_ptr[tid];

    // // --- STEP 3: Local summation (Vertical Reduction) --- 
    // int tile_base = tid * CSR5_OMEGA * CSR5_SIGMA;

    // // Iterate ver sigma 0..15
    // int current_row_offset = y_offset;

    // for (int j=0; j < CSR5_SIGMA; j++) {
    //     int idx = tile_base + lane_id * CSR5_SIGMA + j;
    //     // Read this tranposed data (Coalesced access!) NOTE:
    //     double v = val[idx];
    //     int c = col_idx[idx];
    //     // Check bit flag to see if this is a new row (or even valid)
    //     bool is_new_row = (bit_flag >> (CSR5_SIGMA -1 -j)) & 1;
    //     if (is_new_row) {
    //         current_row_offset++;
    //     }
    //     // Calculate Global Row Index
    //     int global_row = tile_row_start + current_row_offset;
    //     // TODO !!!!!!!!!! SIMPLIFICATION !!!!!!!!!!
    //     // ATOMIC ADD instead of 1000 lines of scanning and stuff I DO NOT UNDERSTAND!
    //     if (v != 0.0) { // the least we can do is skip zeros (STRESS)
    //         atomicAdd(&y[global_row], v * x[c]);
    //     }
    // }
}

extern "C" void spmv_gpu_csr5(
    int m,
    int num_tiles,
    double* d_val,
    int* d_col_idx,
    int* d_row_idx, // Row versioning
    int* d_tile_ptr,
    unsigned int* d_tile_desc,
    double* d_x,
    double* d_y,
    float* time_ms
) {
    dim3 block(CSR5_OMEGA);
    dim3 grid(num_tiles);

    // timers
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    // Reset the output vector since we do atomic adds
    // Reason: we may have multiple threads writing to the same output
    cudaMemset(d_y, 0, m *sizeof(double));
    spmv_csr5_kernel<<<grid, block>>>(num_tiles, d_val, d_col_idx, d_row_idx, d_tile_ptr, d_tile_desc, d_x, d_y);
    checkCudaErrors(cudaEventRecord(start, 0));
    for(int i=0; i<MAX_ITER; i++) {
        spmv_csr5_kernel<<<grid, block>>>(num_tiles, d_val, d_col_idx, d_row_idx, d_tile_ptr, d_tile_desc, d_x, d_y);
    }
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    *time_ms = elapsedTime / (float)MAX_ITER;

    cudaMemset(d_y, 0, m *sizeof(double)); // reset output again after timing Final clean run for correctenss
    spmv_csr5_kernel<<<grid, block>>>(num_tiles, d_val, d_col_idx, d_row_idx, d_tile_ptr, d_tile_desc, d_x, d_y);
    
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaGetLastError());
}
     


