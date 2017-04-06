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
#include <mpi.h>

unsigned long my_ftime() {
    struct timeval t;

    gettimeofday(&t, NULL);
    return (long) (t.tv_sec) * 1000 + (long) (t.tv_usec / 1000);
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
    matrix = (double *) malloc(n * n * sizeof(double));

    if (matrix == NULL) {
        fprintf(stderr, "** Error in matrix creation: insufficient memory **");
        fprintf(stderr, "** Program aborted................................ **");
        exit(1);
        //return (NULL);
    }
    if (random == 0)
        for (i = 0; i < size; ++i)
            for (j = 0; j < m; j++)
                matrix[i * size + j] = 0;
    else if (random == -1)
        for (i = 0; i < size; ++i)
            for (j = 0; j < m; j++)
                matrix[i * size + j] = i + 1;// rand() % 10;  // Initialize with a random value between 0 and 9
    else if (random != -2)
        for (i = 0; i < size; ++i)
            for (j = 0; j < m; j++) matrix[i * size + j] = rand() * (i + j);

    return matrix;   // returns the pointer to the matrix.
}

//-- ----------------------------------------------------------------------------*/
// Display 'size'x'size' elements of 'matrix' for debugging purposes
void display(double *matrix, int size, int atmost) {
    int i, j, limit = size < atmost ? size : atmost;
    for (i = 0; i < limit; i++) {
        for (j = 0; j < limit; j++)
            fprintf(stderr, "%6.1lf ", matrix[i * size + j]);
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
int main(int argc, char *argv[]) {
    int size, debug = 0, taskid, numtasks, from, to;
    unsigned long start_time_lt, initTime, compTime;
    double start, finish;
    char *resultFileName = NULL;
    register int i, j, k;
    register double *A;
    register double *B;
    register double *C;

    // rudimentary argument collecting
    for (k = 1; k < argc; ++k) {
        if (isdigit(argv[k][0]))    // assume it is the matrix size
            size = atoi(argv[k]);
        else if (strncmp(argv[k], "dump", 4) == 0) // assume dump=filename
            resultFileName = strchr(argv[k], '=');
        else if (debug = strncmp(argv[k], "debug", 5) == 0) // want debuging info
            fprintf(stderr, "debug is now on.\n");
    }


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid); /* who am i */
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks); /* number of processors */

    if (size % numtasks != 0) {
        if (taskid == 0) printf("Matrix size not divisible by number of processors\n");
        MPI_Finalize();
        exit(-1);
    }

    //printf("taskid : %d\n", taskid);
    from = taskid * size / numtasks;
    to = (taskid + 1) * size / numtasks;

    if (taskid == 0 && debug) fprintf(stderr, "\nStart parallel MPI/OpenMP algorithm (size=%d)...\n", size);

    if (taskid == 0)
        start_time_lt = my_ftime();  //-- -------------------------- Take starting Time

    A = allocate_real_matrix(size, -1);
    B = allocate_real_matrix(size, -1);
    C = allocate_real_matrix(size, -2);

    /*if(taskid == 0) {
        A = allocate_real_matrix(size, -1);
        B = allocate_real_matrix(size, -1);
        C = allocate_real_matrix(size, -2);
    } else {
        A = allocate_real_matrix(size, -2);
        B = allocate_real_matrix(size, -2);
        C = allocate_real_matrix(size, -2);
    }*/


    if (taskid == 0 && debug) {
        fprintf(stderr, "Created Matrices A, B and C of size %dx%d\n", size, size);
    }

    if (taskid == 0)
        initTime = my_ftime() - start_time_lt;  //-- ----------------- Measure init. Time

    //start = MPI_Wtime(); /* start timer */

    /*MPI_Bcast (B, size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(taskid == 0)
        MPI_Scatter (A, size * size / numtasks, MPI_DOUBLE, MPI_IN_PLACE, size * size / numtasks, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else
        MPI_Scatter (A, size * size / numtasks, MPI_DOUBLE, A + from * size, size * size / numtasks, MPI_DOUBLE, 0, MPI_COMM_WORLD);*/

    for (i = from; i < to; i++) {
#pragma omp parallel shared(A,B,C) private(i,j,k)
        {
#pragma omp for schedule(static)
        for (j = 0; j < size; j++) {
            C[i * size + j] = 0;
            for (k = 0; k < size; k++)
                C[i * size + j] += A[i * size + k] * B[k * size + j];
        }
        }
    }

    if (taskid == 0)
        MPI_Gather(MPI_IN_PLACE, size * size / numtasks, MPI_DOUBLE, C, size * size / numtasks, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
    else
        MPI_Gather(C + from * size, size * size / numtasks, MPI_DOUBLE, C, size * size / numtasks, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);

    //finish = MPI_Wtime(); /*stop timer*/

    //printf("Parallel Elapsed time : %f seconds\n", finish - start);

    if (taskid == 0)
        compTime = my_ftime() - start_time_lt - initTime; //-- --------Measure computing Time

    if (taskid == 0 && debug) {
        fprintf(stderr, "A[%dx%d]:\n", size, size);
        display(A, size, size < 100 ? size : 100);
    }
    if (taskid == 0 && debug) {
        fprintf(stderr, "B[%dx%d]:\n", size, size);
        display(B, size, size < 100 ? size : 100);
    }

    if (taskid == 0 && debug) {
        fprintf(stderr, "C[%dx%d]=A*B:\n", size, size);
        display(C, size, size < 100 ? size : 100);
    }

    if (taskid == 0) {
        // Print and stores timing results in a file
        printf("Times (init and computing) = %.4g, %.4g sec\n\n", initTime / 1000.0, compTime / 1000.0);
        printf("size=%d\tinitTime=%g\tcomputeTime=%g (%lu min, %lu sec)\n", size, initTime / 1000.0, compTime / 1000.0,
               (compTime / 1000) / 60, (compTime / 1000) % 60);
    }

    // Storage of Results and Parametres in the file resultFileName
    if (resultFileName != NULL && taskid == 0) {
        ++resultFileName;      // strchr points to the '=' sign
        if (debug) fprintf(stderr, "dumping result to %s\n", resultFileName);
        FILE *f = fopen(resultFileName, "w");
        if (f == NULL)
            fprintf(stderr, "\nERROR OPENING result file - no results are saved !!\n");
        else {
            dump(C, size, f);
            fclose(f);
        }
    }
    if (taskid == 0 && debug) fprintf(stderr, "Done!\n");

    MPI_Finalize();
    return 0;
}