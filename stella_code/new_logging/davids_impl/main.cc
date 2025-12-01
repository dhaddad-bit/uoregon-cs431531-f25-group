#include <stdio.h>
#include <cstdlib>
#include <string.h>
#include <assert.h>
#include <omp.h>
#include <math.h>

#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#include "main.h"
#include "spmv.h"
#include "common.h"
#include "csr5.h"

#include <string>

#define MAX_FILENAME 256
#define MAX_NUM_LENGTH 100
#define MAX_ITER 100

#define NUM_TIMERS       9
#define LOAD_TIME        0
#define CONVERT_TIME     1
#define SPMV_TIME        2
#define GPU_ALLOC_TIME   4
#define GPU_SPMV_TIME    5
#define GPU_ELL_TIME     3
#define STORE_TIME       6
#define GPU_SELL_C_TIME  7
#define GPU_CSR5_TIME    8

double calc_diff(int m, const double* r1, const double* r2) {
    double diff = 0.0;
    for (int i=0; i<m; i++) {
        double err = r1[i] - r2[i];
        diff += err * err;
    }
    return sqrt(diff);
}


void log_csv(char* matrix, const char* name, int config, double time_s, double conv_s, int nnz, int m, int n, double err) {
   // GFLOPS = (2 * nnz ops) / (time in seconds) / 10^9
    double gflops = (2.0 * nnz) / (time_s * 1e9);

    // Effective Bandwidth (APproximation) THIS IS WHAT WE CARE ABOUT!
    double bytes = (nnz * 12.0) + (m * 16.0) + (n * 8.0); // CSR: val (8 bytes) + col_idx (4 bytes) + row_ptr (4 bytes) + x (8 bytes) + y (8 bytes)
    double gbps = bytes / (time_s * 1e9);
    fprintf(stdout, "%s,%s,%d,%.9f,%.9f,%.4f,%.4f,%e\n", matrix, name, config, time_s, conv_s, gflops, gbps, err);
}

// --- Sell-C-sigma conversion function --- 
extern "C" void convert_csr_to_sell_c_sigma(
    int m, 
    const unsigned int* csr_row_ptr, 
    const unsigned int* csr_col_ind, 
    const double* csr_vals, 
    int SLICE_THICKNESS, 
    int* m_padded, 
    int* total_nnz, 
    unsigned int** sell_slice_ptr, 
    unsigned int** sell_col_ind, 
    double** sell_vals,
    int** sigma_permutation) {
    // --- Step 1: Compute and sort the permutation map based on row lengths ---
    // Allocate permutation array
    *sigma_permutation = (int*)malloc(sizeof(int) * m);
    for (int i = 0; i < m; i++) (*sigma_permutation)[i] = i;

    // Calculate lengths using a lambda function to avoid complications and sort using std::sort
    // Cite: stack overflow
    std::sort(*sigma_permutation, *sigma_permutation + m,
              [csr_row_ptr](int a, int b) {
                    int len_a = csr_row_ptr[a+1] - csr_row_ptr[a];
                    int len_b = csr_row_ptr[b+1] - csr_row_ptr[b];
                    return len_a > len_b;
              }
    );

    // --- Step 2: Slice information calculation --- 
    int num_slices = (m + SLICE_THICKNESS -1) / SLICE_THICKNESS;
    *m_padded = num_slices * SLICE_THICKNESS;

    std::vector<unsigned int> slice_lengths(num_slices);
    unsigned int padded_nnz = 0;
    for (int s = 0; s < num_slices; s++) {
        unsigned int max_length = 0;
        for (int i=0; i<SLICE_THICKNESS; i++) {
            int row = s * SLICE_THICKNESS + i;
            if (row < m) { // we are within original matrix (unpadded bounds)
                // Added logic for sigma
                int orig_row = (*sigma_permutation)[row];
                // Get length of original row
                unsigned int row_length = csr_row_ptr[orig_row+1] - csr_row_ptr[orig_row];
                max_length = std::max(max_length, row_length);
    
            }
        }
        slice_lengths[s] = max_length;
        padded_nnz += max_length * SLICE_THICKNESS;
    }
    *total_nnz = padded_nnz;

    // --- Step 3: Allocate SELL-C Arrays ---
    // Prefix sum on slice lengths to get the slice_ptr array
    *sell_slice_ptr = (unsigned int*)malloc((sizeof(unsigned int) * (num_slices + 1)));
    assert(*sell_slice_ptr);
    *sell_col_ind = (unsigned int*)malloc(sizeof(unsigned int) * *total_nnz);
    assert(*sell_col_ind);
    *sell_vals = (double*)malloc(sizeof(double) * *total_nnz);
    assert(*sell_vals);

    // --- Step 4: Fill the SELL-C Arrays ---
    unsigned int curr_offset = 0;
    (*sell_slice_ptr)[0] = 0;

    for (int s=0; s<num_slices; s++) {
        unsigned int slice_len = slice_lengths[s];
        // Slice pointer update
        curr_offset += slice_len * SLICE_THICKNESS;
        (*sell_slice_ptr)[s+1] = curr_offset;

        unsigned int slice_start = (*sell_slice_ptr)[s];

        for (int i=0; i<SLICE_THICKNESS; i++) {
            int curr_row = s * SLICE_THICKNESS + i;
            unsigned int row_len = 0;
            unsigned int csr_start = 0;

            if (curr_row < m) { // Bounds check and conversion
                int orig_row = (*sigma_permutation)[curr_row];
                csr_start = csr_row_ptr[orig_row];
                row_len = csr_row_ptr[orig_row+1] - csr_start;
            }
            unsigned int j = 0;
            for(; j<row_len; j++) {
                unsigned int sell_idx = slice_start + (j*SLICE_THICKNESS) + i;
                (*sell_col_ind)[sell_idx] = csr_col_ind[csr_start + j]; 
                (*sell_vals)[sell_idx] = csr_vals[csr_start + j];
            }
            // EXPLICIT PADDING ADDED BECAUSE OF ERRORRS!!!!
            for (; j<slice_len; j++) {
                unsigned int sell_idx = slice_start + (j*SLICE_THICKNESS) + i;
                (*sell_col_ind)[sell_idx] = (unsigned int)(-1); 
                (*sell_vals)[sell_idx] = 0.0;
            }
        }
    }
}

int main(int argc, char** argv) {
    // program info
    usage(argc, argv);


    // Initialize timess
    double timer[NUM_TIMERS];
    double time;
    uint64_t t0;
    for(unsigned int i = 0; i < NUM_TIMERS; i++) {
        timer[i] = 0.0;
    }
    InitTSC();

    // Read the sparse matrix file and get its info first
    char matrixName[MAX_FILENAME];
    strcpy(matrixName, argv[1]);
    int is_symmetric = 0;
    read_info(matrixName, &is_symmetric);

	char mat_name[MAX_FILENAME]; // this is diff - will be turned into just the matrix name without other folders
    int index;
    char* ptr;
    const char sep = '/';
    std::string outNamepp = matrixName;
    index = outNamepp.find_last_of(sep);
    strcpy(mat_name, outNamepp.c_str() + index + 1);

    // Read the sparse matrix and store it in row_ind, col_ind, and val,
    // also known as co-ordinate format (COO).
    int ret;
    MM_typecode matcode;
    int m;
    int n;
    int nnz;
    int *row_ind;
    int *col_ind;
    double *val;
    float time_ms; // Variable to capture kernel GPU time
//    fprintf(stdout, "Matrix file name: %s ... ", matrixName);
    t0 = ReadTSC();
    
    ret = mm_read_mtx_crd(matrixName, &m, &n, &nnz, &row_ind, &col_ind, &val, 
                          &matcode);
    check_mm_ret(ret);
    // expand sparse matrix if symmetric
    if(is_symmetric) {
        expand_symmetry(m, n, &nnz, &row_ind, &col_ind, &val);
    }
    timer[LOAD_TIME] += ElapsedTime(ReadTSC() - t0);

    
    // Convert co-ordinate format to CSR format
    unsigned int* csr_row_ptr = NULL; 
    unsigned int* csr_col_ind = NULL;  
    double* csr_vals = NULL; 
    t0 = ReadTSC();
    convert_coo_to_csr(row_ind, col_ind, val, m, n, nnz,
                       &csr_row_ptr, &csr_col_ind, &csr_vals);
    unsigned int* ell_col_ind = NULL;
    double* ell_vals = NULL;
    int n_new = 0;
    convert_csr_to_ell(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz,
                       &ell_col_ind, &ell_vals, &n_new);
    timer[CONVERT_TIME] += ElapsedTime(ReadTSC() - t0);

    // Load the input vector x 
    char vectorName[MAX_FILENAME];
    strcpy(vectorName, argv[2]);
    double* x;
    int vector_size;
    t0 = ReadTSC();
    read_vector(vectorName, &x, &vector_size);
    timer[LOAD_TIME] += ElapsedTime(ReadTSC() - t0);
    assert(n == vector_size);
//    fprintf(stdout, "file loaded\n");

    // ==========================================
    // Calculate CPU SPMV (benchmark/correctness))
    // ==========================================
    double* bb = (double*) malloc(sizeof(double) * m);
    assert(bb);
//    fprintf(stdout, "Calculating CPU CSR SpMV ... ");
    t0 = ReadTSC();
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        spmv(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, x, bb);
    }
    timer[SPMV_TIME] += ElapsedTime(ReadTSC() - t0);
//    fprintf(stdout, "done\n");

    // ==========================================
    // Execute SPMV
    // ==========================================
//    fprintf(stdout, "Executing GPU CSR SpMV ... \n");
    unsigned int* drp; // row pointer on GPU
    unsigned int* dci; // col index on GPU
    double* dv; // values on GPU
    double* dx; // input x on GPU
    double* db; // result b on GPU
    t0 = ReadTSC();
    allocate_csr_gpu(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, x, &drp,
                     &dci, &dv, &dx, &db);
    timer[GPU_ALLOC_TIME] += ElapsedTime(ReadTSC() - t0);

    // Test different thread counts
    int thread_counts[] = {32, 64, 128, 256};
    int num_tests = 1;
    
    // Temp buffer for verification
    double* h_check = (double*)malloc(sizeof(double) * m);
    for (int i = 0; i < num_tests; i++) {
        int threads = thread_counts[i];

        spmv_gpu(drp, dci, dv, m, n, nnz, dx, db, threads, &time_ms);

	// Verify and log
        get_result_gpu(db, h_check, m);
	cudaDeviceSynchronize();

        double err = calc_diff(m, bb, h_check);
        log_csv(mat_name, "CSR", threads, time_ms / 1000.0, 0.0, nnz, m, n, err);
//        fprintf(stdout, " CSR Threads: %d, Time (ms): %f ms\n", threads, time_ms);
        // Only store the time for the 64-thread run in your timer array
        if (threads == 64) timer[GPU_SPMV_TIME] += time * MAX_ITER; // convert to seconds and multiply by iterations
    };

    // copy data back from the GPU
    double* b = (double*) malloc(sizeof(double) * m);;
    assert(b);
    t0 = ReadTSC();
    get_result_gpu(db, b, m);
    timer[GPU_ALLOC_TIME] += ElapsedTime(ReadTSC() - t0);


    // ==========================================
    // Execute ELL SPMV
    // ==========================================
//    fprintf(stdout, "Executing GPU ELL SpMV ... \n");
    unsigned int* dec; // row pointer on GPU
    double* dev; // col index on GPU
    t0 = ReadTSC();
    allocate_ell_gpu(ell_col_ind, ell_vals, m, n_new, nnz, x, &dec, &dev, &dx,
                     &db);
    timer[GPU_ALLOC_TIME] += ElapsedTime(ReadTSC() - t0);

    for (int i = 0; i < num_tests; i++) {
        int threads = thread_counts[i];
        spmv_gpu_ell(dec, dev, m, n_new, nnz, dx, db, threads, &time_ms);
        // Verify and log
        get_result_gpu(db, h_check, m);
	cudaDeviceSynchronize();

        double err = calc_diff(m, h_check, bb);
        log_csv(mat_name, "ELL", threads, time_ms / 1000.0, 0.0, nnz, m, n, err);

//        fprintf(stdout, " ELL Threads: %d, Time (ms): %f ms\n", threads, time_ms);
        // Only store the time for the 64-thread run in your timer array
        if (threads == 64) {
            timer[GPU_ELL_TIME] += time * MAX_ITER; // convert to seconds and multiply by iterations
        }
    }

    // copy data back from the GPU
    double* be = (double*) malloc(sizeof(double) * m);;
    assert(be);
    t0 = ReadTSC();
    get_result_gpu(db, be, m);

    // --- DEBUGGING CODE START ---
    if (bb == NULL) {
        bb = (double*) malloc(sizeof(double) * m);
        for(unsigned int i = 0; i < MAX_ITER; i++) spmv(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, x, bb);
    }
    
    double ell_diff = 0.0;
    for(int i=0; i<m; i++) {
        double err = be[i] - bb[i];
        ell_diff += err * err;
    }
//    printf("DEBUG: ELL 2-Norm: %e\n", sqrt(ell_diff));
    // --- DEBUGGING CODE END ---

    timer[GPU_ALLOC_TIME] += ElapsedTime(ReadTSC() - t0);

    // ==========================================
    // Execute SELL-C-sigma SPMV
    // ==========================================
//    fprintf(stdout, "Executing GPU SELL-C-Sigma SpMV ... \n");
    // Parameterization for SELL-C (not sigma yet TODO)
    int SLICE_THICKNESS = 32;
    int m_padded_sell = 0; int total_nnz_sell = 0;
    unsigned int *h_sell_slice_ptr = NULL;
    unsigned int *h_sell_col_ind = NULL;
    double *h_sell_vals = NULL;
    int *h_sigma_permutation = NULL; // permutation map
    // Run converson on Host CPU
    t0 = ReadTSC();
    convert_csr_to_sell_c_sigma(
        m, csr_row_ptr, csr_col_ind, csr_vals, SLICE_THICKNESS,
        &m_padded_sell, &total_nnz_sell, &h_sell_slice_ptr,
        &h_sell_col_ind, &h_sell_vals, &h_sigma_permutation);

    int num_slices = m_padded_sell / SLICE_THICKNESS;
    timer[CONVERT_TIME] += ElapsedTime(ReadTSC() - t0);
    // Allocate on GPU
    unsigned int *d_sell_slice_ptr = NULL; 
    unsigned int *d_sell_col_ind = NULL;
    double *d_sell_vals = NULL;
    CopyData(h_sell_slice_ptr, num_slices + 1, sizeof(unsigned int), &d_sell_slice_ptr);
    CopyData(h_sell_col_ind, total_nnz_sell, sizeof(unsigned int), &d_sell_col_ind);
    CopyData(h_sell_vals, total_nnz_sell, sizeof(double), &d_sell_vals);
    // Run SPMV Benchmark Loop
    spmv_gpu_sellc(m, num_slices, SLICE_THICKNESS, 
                    d_sell_slice_ptr, d_sell_col_ind, d_sell_vals,
                    dx, db, &time_ms);
    cudaDeviceSynchronize();

    timer[GPU_SELL_C_TIME] += (time_ms / 1000.0) * MAX_ITER; // convert to seconds and multiply by iterations
//    fprintf(stdout, " SELL-C-Sigma Time (ms): %f ms\n", time_ms);
    // Unscramble and verify correctness
    double* h_y_sigma = (double*) malloc(sizeof(double) * m_padded_sell);
    cudaMemcpy(h_y_sigma, db, m * sizeof(double), cudaMemcpyDeviceToHost);
    double sigma_diff = 0.0;
    for (int i=0; i<m; i++) {
        int original_idx = h_sigma_permutation[i];
        double err = h_y_sigma[i] - bb[original_idx];
        sigma_diff += err * err;
        // Store unscrambled result for final save file
        be[original_idx] = h_y_sigma[i];
    }
    log_csv(mat_name, "SELL-C-Sigma", SLICE_THICKNESS, time_ms / 1000.0, 0.0, nnz, m, n, sqrt(sigma_diff));
//    fprintf(stdout, "2-Norm difference between CSR and SELL-C-Sigma results: %e\n", sigma_diff);

    // Free SELL-C specific memory
    cudaFree(d_sell_slice_ptr);
    cudaFree(d_sell_col_ind);
    cudaFree(d_sell_vals);
    free(h_sell_slice_ptr);
    free(h_sell_col_ind);
    free(h_sell_vals);


    // ==========================================
    // Execute CSR5 SPMV
    // ==========================================
//    fprintf(stdout, "Executing GPU CSR5 SpMV ... \n");
    int csr5_num_tiles;
    double *h_csr5_vals = NULL;
    int* h_csr5_col_idx = NULL;
    int* h_csr5_row_idx = NULL;
    int* h_csr5_tile_ptr = NULL;
    unsigned int* h_csr5_tile_desc = NULL;

    // Host Converesion
    t0 = ReadTSC();
    convert_csr_to_csr5(m, n, nnz, csr_row_ptr, csr_col_ind, csr_vals, 
                        &csr5_num_tiles, &h_csr5_vals, &h_csr5_col_idx, 
                        &h_csr5_row_idx, &h_csr5_tile_ptr, &h_csr5_tile_desc);
    timer[CONVERT_TIME] += ElapsedTime(ReadTSC() - t0);

    // GPU allocation
    double* d_csr5_vals = NULL;
    int* d_csr5_col_idx = NULL;
    int* d_csr5_row_idx = NULL;
    int* d_csr5_tile_ptr = NULL;
    unsigned int* d_csr5_tile_desc = NULL;
    // Capacity MORE HANDWAVING IDK WHAT"S GOING ON HERE EITHER TODO 
    int csr5_capacity = csr5_num_tiles * 32 * 16;
    // Values
    cudaMalloc((void**)&d_csr5_vals, csr5_capacity * sizeof(double));
    cudaMemcpy(d_csr5_vals, h_csr5_vals, csr5_capacity * sizeof(double), cudaMemcpyHostToDevice);
    // Column Indices
    cudaMalloc((void**)&d_csr5_col_idx, csr5_capacity * sizeof(int));
    cudaMemcpy(d_csr5_col_idx, h_csr5_col_idx, csr5_capacity * sizeof(int), cudaMemcpyHostToDevice);
    // Row Indices
    cudaMalloc((void**)&d_csr5_row_idx, csr5_capacity * sizeof(int));
    cudaMemcpy(d_csr5_row_idx, h_csr5_row_idx, csr5_capacity * sizeof(int), cudaMemcpyHostToDevice);
    // Tile Pointers
    cudaMalloc((void**)&d_csr5_tile_ptr, (csr5_num_tiles + 1) * sizeof(int));
    cudaMemcpy(d_csr5_tile_ptr, h_csr5_tile_ptr, (csr5_num_tiles + 1) * sizeof(int), cudaMemcpyHostToDevice);
    // Tile Descriptors
    cudaMalloc((void**)&d_csr5_tile_desc, csr5_num_tiles * 32 * sizeof(unsigned int));
    cudaMemcpy(d_csr5_tile_desc, h_csr5_tile_desc, csr5_num_tiles * 32 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // Run the Benchmark loop
    // Reusing dx and db from previous steps
    spmv_gpu_csr5(m, csr5_num_tiles, d_csr5_vals, 
                  d_csr5_col_idx, d_csr5_row_idx, d_csr5_tile_ptr, 
                  d_csr5_tile_desc, dx, db, &time_ms);
    cudaDeviceSynchronize();

    timer[GPU_CSR5_TIME] += (time_ms / 1000.0) * MAX_ITER; // convert to seconds and multiply by iterations
    // Verify and log
    get_result_gpu(db, h_check, m);
    double csr5_diff = calc_diff(m, h_check, bb);
    log_csv(mat_name, "CSR5", 32, time_ms / 1000.0, 0.0, nnz, m, n, csr5_diff);
//    fprintf(stdout, " CSR5 Time (ms): %f ms\n", time_ms);

//    fprintf(stdout, "2-Norm difference between CSR and CSR5 results: %e\n", csr5_diff);
    // Cleanup for CSR5
    cudaFree(d_csr5_vals);
    cudaFree(d_csr5_col_idx);
    cudaFree(d_csr5_row_idx);
    cudaFree(d_csr5_tile_ptr);
    cudaFree(d_csr5_tile_desc);
    free(h_csr5_vals);
    free(h_csr5_col_idx);
    free(h_csr5_row_idx);
    free(h_csr5_tile_ptr);
    free(h_csr5_tile_desc);
    free(h_check);





    // Calculate correctness
    double* c = (double*) malloc(sizeof(double) * m);
    assert(c);
    for(int i = 0; i < m; i++) {
        c[i] = be[i] - bb[i];
    }
    double norm = dnrm2(m, c, 1);   
//    printf("2-Norm between CPU and GPU answers: %e\n", norm);


    // Store the calculated answer in a file, one element per line.
    char resName[MAX_FILENAME];
    strcpy(resName, argv[3]); 
//    fprintf(stdout, "Result file name: %s ... ", resName);
    t0 = ReadTSC();
    store_result(resName, be, m);
    timer[STORE_TIME] += ElapsedTime(ReadTSC() - t0);
//    fprintf(stdout, "file saved\n");


    // print timer
    // print_time(timer);


    // Free memory
    free(csr_row_ptr);
    free(csr_col_ind);
    free(csr_vals);
    free(b);
    free(bb);
    free(c);
    free(x);
    free(row_ind);
    free(col_ind);
    free(val);

    return 0;
}


/* This function checks the number of input parameters to the program to make 
   sure it is correct. If the number of input parameters is incorrect, it 
   prints out a message on how to properly use the program.
   input parameters:
       int    argc
       char** argv 
   return parameters:
       none
 */
void usage(int argc, char** argv)
{
    if(argc < 4) {
        fprintf(stderr, "usage: %s <mat> <vec> <res>\n", 
                argv[0]);
        exit(EXIT_FAILURE);
    } 
}

/* This function prints out information about a sparse matrix
   input parameters:
       char*       fileName    name of the sparse matrix file
       MM_typecode matcode     matrix information
       int         m           # of rows
       int         n           # of columns
       int         nnz         # of non-zeros
   return paramters:
       none
 */
void print_matrix_info(char* fileName, MM_typecode matcode, 
                       int m, int n, int nnz)
{
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Matrix name:     %s\n", fileName);
    fprintf(stdout, "Matrix size:     %d x %d => %d\n", m, n, nnz);
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Is matrix:       %d\n", mm_is_matrix(matcode));
    fprintf(stdout, "Is sparse:       %d\n", mm_is_sparse(matcode));
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Is complex:      %d\n", mm_is_complex(matcode));
    fprintf(stdout, "Is real:         %d\n", mm_is_real(matcode));
    fprintf(stdout, "Is integer:      %d\n", mm_is_integer(matcode));
    fprintf(stdout, "Is pattern only: %d\n", mm_is_pattern(matcode));
    fprintf(stdout, "-----------------------------------------------------\n");
    fprintf(stdout, "Is general:      %d\n", mm_is_general(matcode));
    fprintf(stdout, "Is symmetric:    %d\n", mm_is_symmetric(matcode));
    fprintf(stdout, "Is skewed:       %d\n", mm_is_skew(matcode));
    fprintf(stdout, "Is hermitian:    %d\n", mm_is_hermitian(matcode));
    fprintf(stdout, "-----------------------------------------------------\n");

}


/* This function checks the return value from the matrix read function, 
   mm_read_mtx_crd(), and provides descriptive information.
   input parameters:
       int ret    return value from the mm_read_mtx_crd() function
   return paramters:
       none
 */
void check_mm_ret(int ret)
{
    switch(ret)
    {
        case MM_COULD_NOT_READ_FILE:
            fprintf(stderr, "Error reading file.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_PREMATURE_EOF:
            fprintf(stderr, "Premature EOF (not enough values in a line).\n");
            exit(EXIT_FAILURE);
            break;
        case MM_NOT_MTX:
            fprintf(stderr, "Not Matrix Market format.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_NO_HEADER:
            fprintf(stderr, "No header information.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_UNSUPPORTED_TYPE:
            fprintf(stderr, "Unsupported type (not a matrix).\n");
            exit(EXIT_FAILURE);
            break;
        case MM_LINE_TOO_LONG:
            fprintf(stderr, "Too many values in a line.\n");
            exit(EXIT_FAILURE);
            break;
        case MM_COULD_NOT_WRITE_FILE:
            fprintf(stderr, "Error writing to a file.\n");
            exit(EXIT_FAILURE);
            break;
        case 0:
            //fprintf(stdout, "file loaded.\n");
            break;
        default:
            fprintf(stdout, "Error - should not be here.\n");
            exit(EXIT_FAILURE);
            break;

    }
}

/* This function reads information about a sparse matrix using the 
   mm_read_banner() function and printsout information using the
   print_matrix_info() function.
   input parameters:
       char*       fileName    name of the sparse matrix file
   return paramters:
       none
 */
void read_info(char* fileName, int* is_sym)
{
    FILE* fp;
    MM_typecode matcode;
    int m;
    int n;
    int nnz;

    if((fp = fopen(fileName, "r")) == NULL) {
        fprintf(stderr, "Error opening file: %s\n", fileName);
        exit(EXIT_FAILURE);
    }

    if(mm_read_banner(fp, &matcode) != 0)
    {
        fprintf(stderr, "Error processing Matrix Market banner.\n");
        exit(EXIT_FAILURE);
    } 

    if(mm_read_mtx_crd_size(fp, &m, &n, &nnz) != 0) {
        fprintf(stderr, "Error reading size.\n");
        exit(EXIT_FAILURE);
    }

    // print_matrix_info(fileName, matcode, m, n, nnz);
    *is_sym = mm_is_symmetric(matcode);

    fclose(fp);
}

/* This function converts a sparse matrix stored in COO format to CSR format.
   input parameters:
       int*	row_ind		list or row indices (per non-zero)
       int*	col_ind		list or col indices (per non-zero)
       double*	val		list or values  (per non-zero)
       int	m		# of rows
       int	n		# of columns
       int	n		# of non-zeros
   output parameters:
       unsigned int** 	csr_row_ptr	pointer to row pointers (per row)
       unsigned int** 	csr_col_ind	pointer to column indices (per non-zero)
       double** 	csr_vals	pointer to values (per non-zero)
   return paramters:
       none
 */
void convert_coo_to_csr(int* row_ind, int* col_ind, double* val, 
                        int m, int n, int nnz,
                        unsigned int** csr_row_ptr, unsigned int** csr_col_ind,
                        double** csr_vals)

{
    // Temporary pointers
    unsigned int* row_ptr_;
    unsigned int* col_ind_;
    double* vals_;
    
    // We now how large the data structures should be
    // csr_row_ptr -> m + 1
    // csr_col_ind -> nnz
    // csr_vals    -> nnz
    row_ptr_ = (unsigned int*) malloc(sizeof(unsigned int) * (m + 1)); 
    assert(row_ptr_);
    col_ind_ = (unsigned int*) malloc(sizeof(unsigned int) * nnz);
    assert(col_ind_);
    vals_ = (double*) malloc(sizeof(double) * nnz);
    assert(vals_);

    // Now determine how many non-zero elements are in each row
    // Use a histogram to do this
    unsigned int* buckets = (unsigned int*) malloc(sizeof(unsigned int) * m);
    assert(buckets);
    memset(buckets, 0, sizeof(unsigned int) * m);

    for(unsigned int i = 0; i < nnz; i++) {
        // row_ind[i] - 1 because index in mtx format starts from 1 (not 0)
        buckets[row_ind[i] - 1]++;
    }

    // Now use a cumulative sum to determine the starting position of each
    // row in csr_col_ind and csr_vals - this information is also what is
    // stored in csr_row_ptr
    for(unsigned int i = 1; i < m; i++) {
        buckets[i] = buckets[i] + buckets[i - 1];
    }
    // Copy this to csr_row_ptr
    row_ptr_[0] = 0; 
    for(unsigned int i = 0; i < m; i++) {
        row_ptr_[i + 1] = buckets[i];
    }

    // We can use row_ptr_ to copy the column indices and vals to the 
    // correct positions in csr_col_ind and csr_vals
    unsigned int* tmp_row_ptr = (unsigned int*) malloc(sizeof(unsigned int) * 
                                                       m);
    assert(tmp_row_ptr);
    memcpy(tmp_row_ptr, row_ptr_, sizeof(unsigned int) * m);

    // Now go through each non-zero and copy it to its appropriate position
    for(unsigned int i = 0; i < nnz; i++)  {
        col_ind_[tmp_row_ptr[row_ind[i] - 1]] = col_ind[i] - 1;
        vals_[tmp_row_ptr[row_ind[i] - 1]] = val[i];
        tmp_row_ptr[row_ind[i] - 1]++;
    }

    // Copy the memory address to the input parameters
    *csr_row_ptr = row_ptr_;
    *csr_col_ind = col_ind_;
    *csr_vals = vals_;

    // Free memory
    free(tmp_row_ptr);
    free(buckets);
}

/* Reads in a vector from file.
   input parameters:
       char*	fileName	name of the file containing the vector
   output parameters:
       double**	vector		pointer to the vector
       int*	vecSize 	pointer to # elements in the vector
   return parameters:
       none
 */
void read_vector(char* fileName, double** vector, int* vecSize)
{
    FILE* fp = fopen(fileName, "r");
    assert(fp);
    char line[MAX_NUM_LENGTH];    
    fgets(line, MAX_NUM_LENGTH, fp);
    fclose(fp);

    unsigned int vector_size = atoi(line);
    double* vector_ = (double*) malloc(sizeof(double) * vector_size);

    fp = fopen(fileName, "r");
    assert(fp); 
    // first read the first line to get the # elements
    fgets(line, MAX_NUM_LENGTH, fp);

    unsigned int index = 0;
    while(fgets(line, MAX_NUM_LENGTH, fp) != NULL) {
        vector_[index] = atof(line); 
        index++;
    }

    fclose(fp);
    assert(index == vector_size);

    *vector = vector_;
    *vecSize = vector_size;
}


/* SpMV function for CSR stored sparse matrix
 */
void spmv(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
          double* csr_vals, int m, int n, int nnz, 
          double* vector_x, double *res)
{
    // first initialize res to 0
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < m; i++) {
        res[i] = 0.0;
    }

    // calculate spmv
    #pragma omp parallel for schedule(static)
    for(unsigned int i = 0; i < m; i++) {
        unsigned int row_begin = csr_row_ptr[i];
        unsigned int row_end = csr_row_ptr[i + 1];
        for(unsigned int j = row_begin; j < row_end; j++) {
            res[i] += csr_vals[j] * vector_x[csr_col_ind[j]]; 
        }
    }
}


/* Save result vector in a file
 */
void store_result(char *fileName, double* res, int m)
{
    FILE* fp = fopen(fileName, "w");
    assert(fp);

    fprintf(fp, "%d\n", m);
    for(int i = 0; i < m; i++) {
        fprintf(fp, "%0.10f\n", res[i]);
    }

    fclose(fp);
}

/* Print timing information 
 */
void print_time(double timer[])
{
    fprintf(stdout, "Module\t\tTime\n");
    fprintf(stdout, "Load\t\t");
    fprintf(stdout, "%f\n", timer[LOAD_TIME]);
    fprintf(stdout, "Convert\t\t");
    fprintf(stdout, "%f\n", timer[CONVERT_TIME]);
    fprintf(stdout, "CPU SpMV\t");
    fprintf(stdout, "%f\n", timer[SPMV_TIME]);
    fprintf(stdout, "GPU Alloc\t");
    fprintf(stdout, "%f\n", timer[GPU_ALLOC_TIME]);
    fprintf(stdout, "GPU SpMV\t");
    fprintf(stdout, "%f\n", timer[GPU_SPMV_TIME]);
    fprintf(stdout, "GPU ELL SpMV\t");
    fprintf(stdout, "%f\n", timer[GPU_ELL_TIME]);
    // --- Sell-c specific timing --- 
    fprintf(stdout, "GPU SELL-C SpMV\t");
    fprintf(stdout, "%f\n", timer[GPU_SELL_C_TIME]);
    // ------------------------------- 
    // --- CSR5 specific timing ---
    fprintf(stdout, "GPU CSR5 SpMV\t");
    fprintf(stdout, "%f\n", timer[GPU_CSR5_TIME]);
    // ---------------------------
    fprintf(stdout, "Store\t\t");
    fprintf(stdout, "%f\n", timer[STORE_TIME]);
}


void expand_symmetry(int m, int n, int* nnz_, int** row_ind, int** col_ind, 
                     double** val)
{
    // fprintf(stdout, "Expanding symmetric matrix ... ");
    int nnz = *nnz_;

    // first, count off-diagonal non-zeros
    int not_diag = 0;
    for(int i = 0; i < nnz; i++) {
        if((*row_ind)[i] != (*col_ind)[i]) {
            not_diag++;
        }
    }

    int* _row_ind = (int*) malloc(sizeof(int) * (nnz + not_diag));
    assert(_row_ind);
    int* _col_ind = (int*) malloc(sizeof(int) * (nnz + not_diag));
    assert(_col_ind);
    double* _val = (double*) malloc(sizeof(double) * (nnz + not_diag));
    assert(_val);

    memcpy(_row_ind, *row_ind, sizeof(int) * nnz);
    memcpy(_col_ind, *col_ind, sizeof(int) * nnz);
    memcpy(_val, *val, sizeof(double) * nnz);
    int index = nnz;
    for(int i = 0; i < nnz; i++) {
        if((*row_ind)[i] != (*col_ind)[i]) {
            _row_ind[index] = (*col_ind)[i];
            _col_ind[index] = (*row_ind)[i];
            _val[index] = (*val)[i];
            index++;
        }
    }
    assert(index == (nnz + not_diag));

    free(*row_ind);
    free(*col_ind);
    free(*val);

    *row_ind = _row_ind;
    *col_ind = _col_ind;
    *val = _val;
    *nnz_ = nnz + not_diag;

    //fprintf(stdout, "done\n");
    //fprintf(stdout, "  Total # of non-zeros is %d\n", nnz + not_diag);
}


double ddot(const int n, double* x, const int incx, double* y, const int incy)
{
    double sum = 0.0;
    int max = (n + incx - 1) / incx;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for(int i = 0; i < max; i++) {
        sum += x[i * incx] * y[i * incy];
    }
    return sum;
}

double dnrm2(const int n, double* x, const int incx)
{
    double nrm = ddot(n, x, incx, x, incx);
    return sqrt(nrm);
}


void convert_csr_to_ell(unsigned int* csr_row_ptr, unsigned int* csr_col_ind,
                        double* csr_vals, int m, int n, int nnz, 
                        unsigned int** ell_col_ind, double** ell_vals, 
                        int* n_new)
{
    // --- Find the max number of non-zeros per row (max_row_length) --- 
    int max_row_length = 0;
    for (int i=0; i<m; i++) {
        int row_length =csr_row_ptr[i+1] - csr_row_ptr[i];
        if (row_length > max_row_length) {
            max_row_length = row_length;
        }
    }
    *n_new = max_row_length; // from now on this is the length of our rows
    int ell_nnz_padded = m * max_row_length;
    
    // --- Allocate memory for ELL arrays on the host (CPU) --- 
    *ell_col_ind = (unsigned int*) malloc(sizeof(unsigned int) * ell_nnz_padded);
    assert(*ell_col_ind);
    *ell_vals = (double*) malloc(sizeof(double) * ell_nnz_padded);
    assert(*ell_vals);

    // --- Initialize ELL arrays to zero for padding use (-1) for col/row and 0 for vals --- 
    // Can this be done with calloc or memset? Parallelizable with omp? TODO
    for (int i=0; i<ell_nnz_padded; i++) {
        (*ell_col_ind)[i] = -1; // UNSIGNED INT ISSUE?;
        (*ell_vals)[i] = 0.0;
    }
        
    // --- Fill in ELL arrays from CSR arrays --- 
    // (See slides and paper for details)
    #pragma omp parallel for schedule(static)
    for (int i=0; i<m; ++i) { // "for each row"
        int row_start = csr_row_ptr[i];
        int row_end = csr_row_ptr[i+1];
        int c = 0;
        // Copy data from CSR arrays to ELL arrays
        for (int j=row_start; j<row_end; ++j) {
            int ell_dest_index = i*max_row_length + c; // destination_index = row_index * max_row_length + col_index
            (*ell_col_ind)[ell_dest_index] = csr_col_ind[j];
            (*ell_vals)[ell_dest_index] = csr_vals[j];
            c++;
        }
        // Moved padding initialization above loop to -1/0.0 values.
    }
        // COMPLETE THIS FUNCTION
}



// Code from the third homework for cpu calculation of CSR and COO
void spmv_coo_cpu(unsigned int* row_ind, unsigned int* col_ind, double* vals, 
              int m, int n, int nnz, double* vector_x, double *res, 
              omp_lock_t* writelock)
{
    #pragma omp parallel for
    for (int i=0; i<m; i++) {
        res[i] = 0.0;
    }

    // SpMV Calculation (parallelizable)
    // each thread processes a section of non-zero elts
    #pragma omp parallel for
    for (int i=0; i<nnz; i++) {
        // 1-based to 0-based errors my gosh
        int row = row_ind[i]-1;
        int col = col_ind[i]-1;
        // MUST PREVENT RACE CONDITION updating res[row]
        // Attomic version first draft (less complex/less efficient)
        // #pragma omp atomic
        // res[row] += vals[i] * vector_x[col];
        omp_set_lock(&(writelock[row]));
        res[row] += vals[i] * vector_x[col];
        omp_unset_lock(&(writelock[row]));
    }
}

void spmv_coo_ser_cpu(unsigned int* row_ind, unsigned int* col_ind, double* vals, 
                  int m, int n, int nnz, double* vector_x, double *res)
{
    // Serial version for COO SpMV
    // Initialize result vector to zero
    for (int i=0; i<m; i++) {
        res[i] = 0.0;
    }
    // Cacluate SpMV
    for (int i=0; i<nnz; i++) {
        int row = row_ind[i]-1;
        int col = col_ind[i]-1;
        res[row] += vals[i] * vector_x[col];
    }
}

/* SpMV function for CSR stored sparse matrix
 */
void spmv_ser_cpu(unsigned int* csr_row_ptr, unsigned int* csr_col_ind, 
              double* csr_vals, int m, int n, int nnz, 
              double* vector_x, double *res)
{
    // Serial vereion for SpMV CSR format
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        for (int j = csr_row_ptr[i]; j<csr_row_ptr[i+1]; j++) {
            sum += csr_vals[j] * vector_x[csr_col_ind[j]];
        }
        res[i] = sum;
    }
}

