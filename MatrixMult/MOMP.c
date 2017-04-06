//-- ----------------------------------------------------------------------------*/
//  Standard multiplication algorithm 
//  Author : Pierre Kuonen
//  Created 20.2.2016
//  Modification: Francois Kilchoer 2016/03/04

#define sizeMatrix 2048

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>
#include <limits.h>  // CLOCKS_PER_SEC
#include <sys/timeb.h>
#include <omp.h>

unsigned long my_ftime() {
    struct timeval t;

    gettimeofday(&t, NULL);
    return (long)(t.tv_sec)*1000 + (long)(t.tv_usec/1000);
}

//-- ----------------------------------------------------------------------------
// This function unallocated the matrix (frees memory)
double *free_real_matrix(double *matrix, int size) {
    if (matrix == NULL) return NULL;

#if 0
    for (int i = 0; i < size; i++)
      if (matrix[i]) {
         free(matrix[i]);  // frees a row of the matrix
         matrix[i] = NULL;
      }
#endif
    free(matrix);       // frees the pointer /
    return NULL;        //returns a null pointer /
}

//-- ----------------------------------------------------------------------------
// This function allocates a square matrix using malloc, and initializes it.
// if 'random' is 0, initialize matrix to all 0s
// if 'random' is -1, it initializes the matrix with random values (0->9).
// if 'random' is -2 the matrix is initialized with no values in it.
// else the matrix is initialized with the values of 'random' parameter.
// The parameter 'size' defines the size of the matrix.
double *allocate_real_matrix(int size, int random) {
    int n = size, m = size, i, j;
    double *matrix; //, a;     // pointer to the vector

    // allocates one vector of vectors (2D array = matrix)
    //matrix = (double**) malloc(n * sizeof(double*));
    matrix = (double*) malloc(n*n * sizeof(double));

    if (matrix == NULL) {
        fprintf (stderr, "** Error in matrix creation: insufficient memory **");
        fprintf (stderr, "** Program aborted................................ **");
        exit(1) ;
        //return (NULL);
    }
    if (random == 0)
        for (i=0; i < size; ++i)
            for (j = 0; j < m; j++)
                matrix[i*size+j] = 0;
    else if (random == -1)
        for (i=0; i < size; ++i)
            for (j = 0; j < m; j++)
                matrix[i*size+j] = i+1;// rand() % 10;  // Initialize with a random value between 0 and 9
    else
    if (random != -2)
        for (i=0; i < size; ++i)
            for (j = 0; j < m; j++)  matrix[i*size+j] = rand() * (i+j);

    return matrix;   // returns the pointer to the matrix.
}

//-- ----------------------------------------------------------------------------*/
// Display 'size'x'size' elements of 'matrix' for debugging purposes
void display(double *matrix, int size, int atmost) {
    int i, j, limit = size < atmost ? size : atmost;
    for (i=0; i<limit; i++) {
        for (j=0; j<limit; j++)
            fprintf(stderr, "%6.1lf ", matrix[i*size+j]);
        //printf("%6.1lf ", (double)(*matrix+(i*size+j)));
        if (atmost < size) fprintf(stderr, "...");
        fprintf(stderr, "\n");
    }
    if (atmost < size) fprintf(stderr, "   .... .... ...\n");
    fprintf(stderr, "\n-----------------------\n");
}

// -- --------------------------------------------------------------------
// dump matrix in binary format - simply to allow quick compare with another result file
void dump(double *matrix, int size, FILE *f) {
    if (fwrite(matrix, sizeof(double), size, f) != size)
        fprintf(stderr, "Couldn't dump results to file\n");
}

// -- -----------------------------------------------------------
int main (int argc, char* argv[]) {
    int size, debug=0;
    unsigned long start_time_lt, initTime, compTime;
    char* resultFileName = NULL;
    register int i, j, k;

    // rudimentary argument collecting
    for (k=1; k < argc; ++k) {
        if (isdigit(argv[k][0]))    // assume it is the matrix size
            size = atoi(argv[k]);
        else if (strncmp(argv[k], "dump", 4) == 0) // assume dump=filename
            resultFileName=strchr(argv[k],'=');
        else if (debug=strncmp(argv[k], "debug", 5) == 0) // want debuging info
            fprintf(stderr, "debug is now on.\n");
    }

    if (debug) fprintf(stderr, "\nStart sequential standard algorithm (size=%d)...\n", size);

    start_time_lt =  my_ftime();  //-- -------------------------- Take starting Time

    register double* A=allocate_real_matrix(size, -1);
    register double* B=allocate_real_matrix(size, -1);
    register double* C=allocate_real_matrix(size, -2);

    if (debug) fprintf(stderr, "Created Matrices A, B and C of size %dx%d\n", size, size);

    initTime = my_ftime() - start_time_lt;  //-- ----------------- Measure init. Time

    #pragma omp parallel shared(A,B,C) private(i,j,k)
    {
        #pragma omp for schedule(static)
        for (i = 0; i < size; i++) {
            for (j = 0; j < size; j++) {
                C[i * size + j] = 0;
                for (k = 0; k < size; k++) {
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
                }
            }
        }
    }

    compTime = my_ftime() - start_time_lt - initTime; //-- --------Measure computing Time
    if (debug) { fprintf(stderr, "A[%dx%d]:\n", size, size); display(A, size, size < 100? size:100); }
    if (debug) { fprintf(stderr, "B[%dx%d]:\n", size, size); display(B, size, size < 100? size:100); }


    // Print and stores timing results in a file
    printf("Times (init and computing) = %.4g, %.4g sec\n\n",
           initTime/1000.0, compTime/1000.0);
    printf("size=%d\tinitTime=%g\tcomputeTime=%g (%lu min, %lu sec)\n", size, initTime/1000.0, compTime/1000.0,
           (compTime/1000)/60, (compTime/1000)%60);
    if (debug) { fprintf(stderr, "C[%dx%d]=A*B:\n", size, size); display(C, size, size < 100? size:100);}

    // Storage of Results and Parametres in the file resultFileName
    if (resultFileName!=NULL) {
        ++resultFileName;      // strchr points to the '=' sign
        if (debug) fprintf(stderr, "dumping result to %s\n", resultFileName);
        FILE* f = fopen(resultFileName, "w");
        if (f == NULL)
            fprintf(stderr, "\nERROR OPENING result file - no results are saved !!\n");
        else {
            dump(C, size, f);
            fclose(f);
        }
    }
    if (debug) fprintf(stderr, "Done!\n");
    return 0;
}