#include "csr5.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstring>

// Helper: Binary search finds which row an index belongs to
int binary_search(const unsigned int* row_ptr, int val, int num_rows) {
    int l = 0, r = num_rows -1;
    int res = -1;
    while (l <= r) {
        int mid = l + (r -l) / 2;
        if (row_ptr[mid] <= val) {
            res = mid;
            l = mid + 1;
        } else {
            r = mid -1;
        }
    }
    // row index where row_ptr[i] <= val < row_ptr[i+1]
    // upper_bound logic from std:: gives us this functionality in C++ (stack overflow + gemini)
    const unsigned int* it = std::upper_bound(row_ptr, row_ptr + num_rows + 1, val);
    return (int)(it - row_ptr) -1;
}

// Helper: Set Bits!!! (from CSR5 paper)
int popcount(uint32_t x) {
    return __builtin_popcount(x);
}
// I Had to rename a bunch of variables here to make them flow better
// Essentially we use the csr5..._out variables to return the output arrays
// But we work with the local variables csr5... inside the function.
// There's gotta be a cleaner way but im too tired to think about it rn
extern "C" void convert_csr_to_csr5(
    int m, int n, int nnz,
    const unsigned int* h_row_ptr,
    const unsigned int* h_col_idx,
    const P_TYPE* h_val,
    int* num_tiles,
    P_TYPE** h_csr5_val,
    int** h_csr5_col_idx,
    int** h_csr5_row_idx, // Row Versioning
    int** h_csr5_tile_ptr,
    uint32_t** h_csr5_tile_desc
) {
    // --- STEP 1: Calculate the number of tiles --- 
    const int OMEGA = CSR5_OMEGA;
    const int SIGMA = CSR5_SIGMA;
    int num_tiles_local = (nnz + OMEGA * SIGMA -1) / (OMEGA * SIGMA);
    *num_tiles = num_tiles_local;
    
    // --- STEP 2: Allocate output arrays --- 
    int capacity = (*num_tiles *(OMEGA * SIGMA));
    P_TYPE *csr5_val = (P_TYPE*) calloc (capacity, sizeof(P_TYPE)); 
    int* csr5_col_idx =(int*) calloc (capacity, sizeof(int));
    int* csr5_row_idx =(int*) calloc (capacity, sizeof(int));
    int* tile_ptr = (int*) malloc((num_tiles_local +1) * sizeof(int));
    unsigned int* tile_desc = (unsigned int*) calloc (num_tiles_local * OMEGA, sizeof(unsigned int));

    // Temporary vectors for tile descriptor construction
    std::vector<unsigned int> bit_flag(OMEGA * SIGMA);
    std::vector<int> y_offset(OMEGA);
    std::vector<int> seg_offset(OMEGA);

    // --- STEP 3: Iterate over each tile and fill in the data to the CSR5 arrays --- 
    for (int tile_id = 0; tile_id < num_tiles_local; tile_id++) {
        int tile_start = tile_id * OMEGA * SIGMA; // first element of tile (global index)
        
        // 3.1 Tile Pointer
        int row_idx = binary_search(h_row_ptr, tile_start, m);
        if (h_row_ptr[row_idx] == (unsigned int)tile_start) {
            row_idx--;
        }
        tile_ptr[tile_id] = row_idx;

        // 3.2 Transpose + Generate Bit Flags
        std::fill(bit_flag.begin(), bit_flag.end(), 0);
        for (int i=0; i<OMEGA; i++) {
            for (int j=0; j< SIGMA; j++) {
                int current_idx = tile_start + j * OMEGA + i;
                int dest_idx = tile_start + i * SIGMA + j;

                if (current_idx < nnz) {
                    csr5_val[dest_idx] = h_val[current_idx];
                    csr5_col_idx[dest_idx] = h_col_idx[current_idx];
                    // Row Versioning
                    int r = binary_search(h_row_ptr, current_idx, m);
                    csr5_row_idx[dest_idx] = r;

                    // // Set Bit Flag
                    // int r = binary_search(h_row_ptr, current_idx, m);
                    // if (h_row_ptr[r] == current_idx) {
                    //     // Check if this is the first element of a row
                    //     bit_flag[i * SIGMA + j] = 1;
                    // }
                } else {
                    csr5_row_idx[dest_idx] = -1; // Row Versioning: Mark as invalid
                }
            }
        }
        // 3.3 Generate y_offset (prefix sum of bit flags easy peasy)
        // Could parallelize this later TODO OMFG
        int running_offset = 0;
        for (int i=0; i<OMEGA; i++) {
            int col_count = 0; // Reason: Count number of set bits in this column (carmen had good image for this)
            for (int j=0; j<SIGMA; j++) {
                if (bit_flag[i*SIGMA + j]) {
                    col_count++;
                }
            }
            y_offset[i] = running_offset;
            running_offset += col_count;
        }

        // 3.4 Generate seg_offset
        // For each column, we look for segments (set bits)
        for (int i=0; i<OMEGA; i++) {
            int distance = 0;
            // Forward scan of columns
            for (int c=i+1; c<OMEGA; c++) {
                bool has_bit_flag = false;
                for (int j=0; j<SIGMA; j++) {
                    if (bit_flag[c*SIGMA + j]) {
                        has_bit_flag = true;
                        break;
                    }
                }
                if (has_bit_flag) break; // Found a segment
                distance++; // No segment found yet
            }
            seg_offset[i] = distance;
        }

        // 3.5 Pack tile_desc based on y_offset and seg_offset
        // [ bit_flag (SIGMA bits) | y_offset (ceil(log2(SIGMA*OMEGA))) | seg_offset (ceil(log2(OMEGA))) ] 
        // Explanation: Each tile_desc entry corresponds to a column in the tile
        // We pack the bit flags, y_offset, and seg_offset into a single 32-bit integer (like the paper did)
        // Note: We assume SIGMA*OMEGA fits within 32 bits?? I don't understand this yet
        // TODO
        // In our case [ bit_flag (16) | y_offset (9) | seg_offset (7) ]
        for (int i=0; i<OMEGA; i++) {
            unsigned int desc = 0;
            // Pack bit_flag (Highest bits) (SIGMA BITS)
            unsigned int bf = 0;
            for (int j=0; j<SIGMA; j++) {
                if (bit_flag[i*SIGMA + j]) {
                    bf |= (1 << (SIGMA - 1 - j));
                }
            }
            desc |= (bf << 16);
            // Pack y_offset (Middle bits) (log2(SIGMA*OMEGA) BITS)
            desc |= ((y_offset[i] & 0x1FF) << 7); // 16bits 1111 1111 1000 0000 
            // Pack seg_offset (Loweset bits) (log2(OMEGA) BITS)
            desc |= (seg_offset[i] & 0x7F); // 16bits 9bits 0111 1111 

            tile_desc[tile_id*OMEGA + i] = desc;
        }
    }
    // The final tile_ptr entry is end of last row = m
    tile_ptr[*num_tiles] = m;
    
    // Assign the outputs to pointers
    *h_csr5_val = csr5_val;
    *h_csr5_col_idx = csr5_col_idx;
    *h_csr5_row_idx = csr5_row_idx;
    *h_csr5_tile_ptr = tile_ptr;
    *h_csr5_tile_desc = tile_desc;
}