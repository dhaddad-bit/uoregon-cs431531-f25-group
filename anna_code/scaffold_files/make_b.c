#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// for simplicity's sake, this only generates doubles between 0 and 1
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s {number of rows} {name}\n", argv[0]);
        return 1;
    }
    
    int rows = atoi(argv[1]);
    srand((unsigned)time(NULL));

    FILE *fptr;
    char *name = argv[2];
    
    char *result = malloc(strlen("b_") + strlen(name) + strlen(".mtx") + 1);

    strcpy(result, "b_");
    strcat(result, name);
    strcat(result, ".mtx");

    fptr = fopen(result, "w");
    if (fptr == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    

    fprintf(fptr, "%d\n", rows);

    for (int i = 0; i < rows; i++) {
        double rand_num = (double)rand() / (double)RAND_MAX;
        fprintf(fptr, "%.8f\n", rand_num);
    }

    fclose(fptr);

    free(result);

    return 0;
}