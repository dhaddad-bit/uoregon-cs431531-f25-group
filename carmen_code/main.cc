#include <stdio.h>
//#include <stdlib.h>
#include <cstdlib>
#include <cstring>
#include <assert.h>
#include <omp.h>
#include <math.h>

#include "main.h"
#include "spmv.h"
#include "csr5.h"

#define MAX_FILENAME 256
#define MAX_NUM_LENGTH 100
#define MAX_ITER 10

#define NUM_TIMERS       7
#define LOAD_TIME        0
#define CONVERT_TIME     1
#define SPMV_TIME        2
#define GPU_ALLOC_TIME   4
#define GPU_SPMV_TIME    5
#define GPU_ELL_TIME     3
#define STORE_TIME       6



int main(int argc, char** argv)
{
    // program info
    usage(argc, argv);


    // Initialize timess
    double timer[NUM_TIMERS];
    uint64_t t0;
    for(unsigned int i = 0; i < NUM_TIMERS; i++) {
        timer[i] = 0.0;
    }
    InitTSC();


    // get CG parameters
	/*
    int max_iter = atoi(argv[4]);
    double tol = atof(argv[5]);
	 */


    // Read the sparse matrix file and get its info first
    char matrixName[MAX_FILENAME];
    strcpy(matrixName, argv[1]);
    int is_symmetric = 0;
    read_info(matrixName, &is_symmetric);


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
    fprintf(stdout, "Matrix file name: %s ... ", matrixName);
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
    fprintf(stdout, "Converting COO to CSR...");
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
    fprintf(stdout, "done\n");
    timer[CONVERT_TIME] += ElapsedTime(ReadTSC() - t0);

    // Load the input vector x 
    char vectorName[MAX_FILENAME];
    strcpy(vectorName, argv[2]);
    fprintf(stdout, "Vector file name: %s ... ", vectorName);
    double* x;
    int vector_size;
    t0 = ReadTSC();
    read_vector(vectorName, &x, &vector_size);
    timer[LOAD_TIME] += ElapsedTime(ReadTSC() - t0);
    assert(n == vector_size);
    fprintf(stdout, "file loaded\n");


    // Execute SPMV
    fprintf(stdout, "Executing GPU CSR SpMV ... \n");
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
    int num_tests = 4;

    for (int i = 0; i < num_tests; i++) {
        int threads = thread_counts[i];
        fprintf(stdout, "  Running CSR with %d threads...\n", threads);
        t0 = ReadTSC();
        spmv_gpu(drp, dci, dv, m, n, nnz, dx, db, threads);

        // Only store the time for the 64-thread run in your timer array
        if (threads == 64) {
            timer[GPU_SPMV_TIME] += ElapsedTime(ReadTSC() - t0);
        }
    };

    // copy data back from the GPU
    double* b = (double*) malloc(sizeof(double) * m);;
    assert(b);
    t0 = ReadTSC();
    get_result_gpu(db, b, m);
    timer[GPU_ALLOC_TIME] += ElapsedTime(ReadTSC() - t0);


    // Execute ELL SPMV
    fprintf(stdout, "Executing GPU ELL SpMV ... \n");
    unsigned int* dec; // row pointer on GPU
    double* dev; // col index on GPU
    t0 = ReadTSC();
    allocate_ell_gpu(ell_col_ind, ell_vals, m, n_new, nnz, x, &dec, &dev, &dx,
                     &db);
    timer[GPU_ALLOC_TIME] += ElapsedTime(ReadTSC() - t0);

    for (int i = 0; i < num_tests; i++) {
        int threads = thread_counts[i];
        fprintf(stdout, "  Running ELL with %d threads...\n", threads);
        t0 = ReadTSC();
        spmv_gpu_ell(dec, dev, m, n_new, nnz, dx, db, threads);

        // Only store the time for the 64-thread run in your timer array
        if (threads == 64) {
            timer[GPU_ELL_TIME] += ElapsedTime(ReadTSC() - t0);
        }
    }

    // copy data back from the GPU
    double* be = (double*) malloc(sizeof(double) * m);;
    assert(be);
    t0 = ReadTSC();
    get_result_gpu(db, be, m);
    timer[GPU_ALLOC_TIME] += ElapsedTime(ReadTSC() - t0);



    // Calculate CPU SPMV
    double* bb = (double*) malloc(sizeof(double) * m);
    assert(bb);
    fprintf(stdout, "Calculating CPU CSR SpMV ... ");
    t0 = ReadTSC();
    for(unsigned int i = 0; i < MAX_ITER; i++) {
        csr_spmv(csr_row_ptr, csr_col_ind, csr_vals, m, n, nnz, x, bb);
    }
    timer[SPMV_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "done\n");


    // Calculate correctness
    double* c = (double*) malloc(sizeof(double) * m);
    assert(c);
    for(int i = 0; i < m; i++) {
        c[i] = be[i] - bb[i];
    }
    double norm = dnrm2(m, c, 1);   
    printf("2-Norm between CPU and GPU answers: %e\n", norm);


    // Store the calculated answer in a file, one element per line.
    char resName[MAX_FILENAME];
    strcpy(resName, argv[3]); 
    fprintf(stdout, "Result file name: %s ... ", resName);
    t0 = ReadTSC();
    store_result(resName, be, m);
    timer[STORE_TIME] += ElapsedTime(ReadTSC() - t0);
    fprintf(stdout, "file saved\n");


    // print timer
    print_time(timer);


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
            fprintf(stdout, "file loaded.\n");
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

    print_matrix_info(fileName, matcode, m, n, nnz);
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
    fprintf(stdout, "Store\t\t");
    fprintf(stdout, "%f\n", timer[STORE_TIME]);
}


void expand_symmetry(int m, int n, int* nnz_, int** row_ind, int** col_ind, 
                     double** val)
{
    fprintf(stdout, "Expanding symmetric matrix ... ");
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

    fprintf(stdout, "done\n");
    fprintf(stdout, "  Total # of non-zeros is %d\n", nnz + not_diag);
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


