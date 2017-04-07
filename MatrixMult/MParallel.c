//-- ----------------------------------------------------------------------------*/
//  Parallel multiplication algorithm (with MPI and OpenMP)
//  Author : Pierre Kuonen
//  Created 20.2.2016
//  Modification: Jérôme Moret & Dousse Kewin 06.04.2017

#define sizeMatrix 2048

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
    int size, debug = 0, taskid, numtasks;
    unsigned long start_time_lt, initTime, compTime, sendTime;
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

    int stripe_height = size / numtasks;
    int extra_stripe_height = size % numtasks; // if size isn't divisible by number of processors

    // Stripe indices
    int from = taskid * stripe_height;
    int to = (taskid + 1) * stripe_height;
    if (taskid == numtasks - 1) // Last worker compute last stripe + extra stripe
        to += extra_stripe_height;

    if (taskid == 0 && debug) fprintf(stderr, "\nStart parallel MPI/OpenMP algorithm (size=%d)...\n", size);

    // Init time = time to allocate and to send matrices to workers
    if (taskid == 0)
        start_time_lt = my_ftime();  //-- -------------------------- Take starting Time

    int stripe_size = stripe_height * size;
    int last_stripe_size = extra_stripe_height * size + stripe_size;

    if (taskid == 0) { // Fill only on master node
        A = allocate_real_matrix(size, -1);
        B = allocate_real_matrix(size, -1);
        C = allocate_real_matrix(size, -2);
    } else {
        A = allocate_real_matrix(size, -2);
        B = allocate_real_matrix(size, -2);
        C = allocate_real_matrix(size, -2);
    }

    // Each node compute size / #nodes lines of C so we need to broadcast full matrix B
    MPI_Bcast(B, size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Send only my concerned stripe
    int *displs = (int *) malloc(numtasks * sizeof(int)); /* displacement (relative to send buffer) */
    int *scounts = (int *) malloc(numtasks * sizeof(int)); /* nb of elements to send */

    // For worker 1 to n - 1
    for (i = 0; i < numtasks - 1; ++i) {
        displs[i] = i * stripe_size;
        scounts[i] = stripe_size;
    }

    // For last worker
    displs[i] = i * stripe_size;
    scounts[i] = last_stripe_size;

    if (taskid == 0)
        initTime = my_ftime() - start_time_lt; //-- ------------------ Measure init. Time

    // Send stripes of A to workers
    if (taskid == 0)
        MPI_Scatterv(A, scounts, displs, MPI_DOUBLE, MPI_IN_PLACE, stripe_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else
        MPI_Scatterv(A, scounts, displs, MPI_DOUBLE, A + from * size,
                     (taskid == numtasks - 1) ? last_stripe_size : stripe_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (taskid == 0)
        sendTime = my_ftime() - start_time_lt;  //-- ----------------- Measure send. Time

    if (taskid == 0 && debug) {
        fprintf(stderr, "Created and sent Matrices A, B and C of size %dx%d\n", size, size);
    }

    // Each node compute the multiplication (MPI)
    // Parallelization of multiplication on a node (OpenMP)
#pragma omp parallel shared(A,B,C) private(i,j,k)
    {
#pragma omp for schedule(static)
        for (i = from; i < to; i++)
            for (j = 0; j < size; j++) {
                C[i * size + j] = 0;
                for (k = 0; k < size; k++)
                    C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
    }

    // Get stripes of C from workers
    if (taskid == 0)
        MPI_Gatherv(MPI_IN_PLACE, stripe_size, MPI_DOUBLE, C, scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    else
        MPI_Gatherv(C + from * size, (taskid == numtasks - 1) ? last_stripe_size : stripe_size, MPI_DOUBLE, C, scounts,
                    displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
        printf("Times (init, send and computing) = %.4g, %.4g, %.4g sec\n\n", initTime / 1000.0, sendTime / 1000.0, compTime / 1000.0);
        printf("size=%d\tinitTime=%g\tsendTime=%g\tcomputeTime=%g (%lu min, %lu sec)\n", size, initTime / 1000.0, sendTime / 1000.0, compTime / 1000.0,
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

    free_real_matrix(A, size);
    free_real_matrix(B, size);
    free_real_matrix(C, size);
    free(scounts);
    free(displs);

    MPI_Finalize();

    return 0;
}
