#include <stdio.h>
//#include <stdlib.h>
#include <cstdlib>
#include <string.h>
#include <assert.h>
#include <omp.h>

#include "mmio.h"
#include "gpu_spmv.h"

#include <algorithm> // for std::sort
#include <vector>

#define MAX_LEN 100

struct COOEntry {
    unsigned int row;
    unsigned int col;
    double val;
};

void read_vector(const char *file, double **vector, int *size) {

    char buffer[MAX_LEN];
    char read[MAX_LEN];
    int s;

    FILE *fptr = fopen(file, "r");
    if (fptr == NULL) {
        perror("Error opening vector file");
    }

    if (fgets(buffer, MAX_LEN, fptr) != NULL) {
        s = atoi(buffer);
    }

    double *new_v = (double *)malloc(sizeof(double) * s);
    int i = 0;
    while (fgets(buffer, MAX_LEN, fptr) != NULL) {
        char *endptr;
        double num = strtod(buffer, &endptr);
        new_v[i++] = num;
    }

    *size = s;
    *vector = new_v;

    fclose(fptr);
}

void read_matrix(const char *file, int *num_rows, int *num_cols, int *num_nnz, 
    unsigned int **rows, unsigned int **cols, double **values) {

    FILE* fp;
    MM_typecode matcode;
    int m, n, nnz;

    if((fp = fopen(file, "r")) == NULL) {
        fprintf(stderr, "Error opening file: %s\n", file);
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

    unsigned int *r_array = (unsigned int *)malloc(sizeof(unsigned int) * nnz);
    unsigned int *c_array = (unsigned int *)malloc(sizeof(unsigned int) * nnz);
    double *nz_array = (double *)malloc(sizeof(double) * nnz);

    if(mm_read_mtx_crd_data(fp, m, n, nnz, (int *)r_array, (int *)c_array, nz_array, matcode) != 0) {
        fprintf(stderr, "Error reading matrix data.\n");
        exit(EXIT_FAILURE);
    }

    if (mm_is_symmetric(matcode)) {

        int actual_nnz = 0;

        for (int i = 0; i < nnz; i++) {
            if (r_array[i] != c_array[i]) {
                actual_nnz += 2;
            } else {
                actual_nnz += 1;
            }
        }

        unsigned int *actual_r = (unsigned int *)malloc(sizeof(unsigned int) * actual_nnz);
        unsigned int *actual_c = (unsigned int *)malloc(sizeof(unsigned int) * actual_nnz);
        double *actual_nnz_arr = (double *)malloc(sizeof(double) * actual_nnz);

        int current = 0;
        for (int i = 0; i < nnz; i++) {
            actual_r[current] = r_array[i];
            actual_c[current] = c_array[i];
            actual_nnz_arr[current] = nz_array[i];
            current++;

            // if an elem is not on diagonal, then (i, j) = (j, i)
            if (r_array[i] != c_array[i]) {
                actual_r[current] = c_array[i];
                actual_c[current] = r_array[i];
                actual_nnz_arr[current] = nz_array[i];
                current++; // written 2 elements
            }
        }

        free(r_array);
        free(c_array);
        free(nz_array);

        r_array = actual_r;
        c_array = actual_c;
        nz_array = actual_nnz_arr;
        nnz = actual_nnz;

    }

    *rows = r_array;
    *cols = c_array;
    *values = nz_array;
    *num_rows = m;
    *num_cols = n;
    *num_nnz = nnz;

    fclose(fp);
}

void convert_coo_to_csr(unsigned int *row_ind, unsigned int *col_ind, double *val, 
                unsigned int **csr_row_ptr, unsigned int **csr_col_ind, double **csr_vals,
                int m, int n, int nnz)
{
    // row ptr array is one larger than number of rows
    int size = m + 1;
    // Allocate all arrays
    *csr_row_ptr = (unsigned int *)malloc(sizeof(unsigned int) * size);
    *csr_col_ind = (unsigned int *)malloc(sizeof(unsigned int) * nnz);
    *csr_vals = (double *)malloc(sizeof(double) * nnz);

    // Set all values in the row array to 0
    for (int index = 0; index < size; index++) {
        (*csr_row_ptr)[index] = 0;
    }

    // Use histogram to store number of values in each row
    for (int index = 0; index < nnz; index++) {
        // Count the number of times row appears in the COO format
        // row_ind[index] = row number so (*csr_row_ptr)[row_ind[index]] = place in row array
        (*csr_row_ptr)[row_ind[index]]++;
    }
    // Use prefix-sum algorithm to get pointer values in row array
    for (int i = 1; i <= m; i++) {
        // Iterate through csr row ptr array
        // Add the previous value to the current one
        (*csr_row_ptr)[i] += (*csr_row_ptr)[i - 1];
    }

    // Make copy of row array and use it to copy indices and values in arrays
    unsigned int *csr_row_copy = (unsigned int *)malloc(sizeof(unsigned int) * size);
    memcpy(csr_row_copy, (*csr_row_ptr), sizeof(unsigned int) * size);

    for (int i = 0; i < nnz; i++) {
        // Get the row number
        int row = row_ind[i] - 1;
        // The place in the row ptr array corresponds to the value in the coo row number
        // so if row_ind[i] = 2, then csr_row_copy[row_ind[i]] will be the place to store 
        // it in col and value arrays
        int place = csr_row_copy[row];
        (*csr_col_ind)[place] = col_ind[i] - 1;
        (*csr_vals)[place] = val[i];
        // We need to increment the number in csr row ptr array to find next spot to place in array
        // for values in the same row
        csr_row_copy[row]++;
    }

    free(csr_row_copy);
}

void spmv_csr(unsigned int *row_ptr, unsigned int *col_ind, double *vals,
     int m, int n, int nnz, double *vector, double *result) {
        #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < m; i++) {
            result[i] = 0.0;
        }
        // Go through rows
        #pragma omp for
        for (int i = 0; i < m; i++) {
            // find pointer to where row starts (number)
            int start = row_ptr[i];
            // end is the next pointer in array
            int end = row_ptr[i+1];
            for (int j = start; j < end; j++) {
                int vector_place = col_ind[j]; // row in vector == column index of matrix
                result[i] += vals[j] * vector[vector_place];
            }
        }
    }
}

void store_result(const char *file, double *result, int m) {
    FILE* fptr = fopen(file, "w");
    assert(fptr);

    fprintf(fptr, "%d\n", m);
    for(int i = 0; i < m; i++) {
        fprintf(fptr, "%0.10f\n", result[i]);
    }

    fclose(fptr);
}

int main(int argc, char *argv[]) {

    if (argc != 4) {
        printf("Usage: ./test_csr {input matrix file} {vector file} {output file}");
        return 1;
    }
    char *input_name = argv[1];
    char *vector_name = argv[2];
    char *output_name = argv[3];

    // read matrix in csr format
    unsigned int *coo_rows_arr;
    unsigned int *coo_cols_arr;
    double *coo_nnz_arr;

    int num_rows = 0;
    int num_cols = 0;
    int nnz = 0;
    read_matrix(input_name, &num_rows, &num_cols, &nnz, &coo_rows_arr, &coo_cols_arr, &coo_nnz_arr);

    unsigned int *csr_rows_arr;
    unsigned int *csr_cols_arr;
    double *csr_nnz_arr;
    convert_coo_to_csr(coo_rows_arr, coo_cols_arr, coo_nnz_arr, &csr_rows_arr,
         &csr_cols_arr, &csr_nnz_arr, num_rows, num_cols, nnz);
    
    double *vector = NULL;
    int vec_size = 0;
    read_vector(vector_name, &vector, &vec_size);

    unsigned int* drp; // row pointer on GPU
    unsigned int* dci; // col index on GPU
    double* dv; // values on GPU
    double* dx; // input x on GPU
    double* db; // result b on GPU

    // // if using vector COO: sort the arrays and change to 0-index
    // std::vector<COOEntry> coo_vec(nnz);
    // for (int i = 0; i < nnz; i++) {
    //     coo_vec[i].row = coo_rows_arr[i] - 1; // 0-based
    //     coo_vec[i].col = coo_cols_arr[i] - 1; // 0-based
    //     coo_vec[i].val = coo_nnz_arr[i];
    // }

    // // sort by row, then by column
    // std::sort(coo_vec.begin(), coo_vec.end(),
    //     [](const COOEntry &a, const COOEntry &b) {
    //         if (a.row == b.row) return a.col < b.col;
    //         return a.row < b.row;
    //     });

    // // copy back to arrays
    // for (int i = 0; i < nnz; i++) {
    //     coo_rows_arr[i] = coo_vec[i].row;
    //     coo_cols_arr[i] = coo_vec[i].col;
    //     coo_nnz_arr[i] = coo_vec[i].val;
    // }

    allocate_csr_gpu(coo_rows_arr, coo_cols_arr, coo_nnz_arr, num_rows, num_cols, nnz, vector, 
                    &drp, &dci, &dv, &dx, &db);

    

    scalar_csr(drp, dci, dv, num_rows, num_cols, nnz, dx, db);
    double* b = (double*) malloc(sizeof(double) * num_rows);
    get_result_gpu(db, b, num_rows);

    store_result(output_name, b, num_rows);

    /* cleanup */
    free(vector);
    free(b);
    free(coo_rows_arr);
    free(coo_cols_arr);
    free(coo_nnz_arr);
    free(csr_rows_arr);
    free(csr_cols_arr);
    free(csr_nnz_arr);

    return 0;
}