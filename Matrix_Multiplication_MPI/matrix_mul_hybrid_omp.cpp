#include <iostream>
#include <cstdlib>
#include <chrono>
#include <string>
#include <mpi.h>   // Distributed memory (communication)
#include <omp.h>   // Shared memory (local threading)

using namespace std;

// Allocates memory for a 1D array of a specified size.
int* allocateMatrix(int size) {
    int* matrix = new int[size]; 
    return matrix;
}

// Frees the memory allocated for a matrix.
void deallocateMatrix(int* matrix) {
    delete[] matrix;
}

// Fills an N*N matrix with random integer values (0-99). Only run on the root (Rank 0).
void initializeMatrix(int* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 100;
    }
}

// Function: localMatrixMultiplication_OMP
// Description: Computes C_local = A_local * B. The outermost loop (row loop 'i') is parallelized using OpenMP threads.
// Parameters: A_local (int*) - Local rows of A, B (int*) - Full B matrix, C_local (int*) - Local result C rows, 
//             N (int) - Matrix dimension, local_M (int) - Number of rows in A_local/C_local.
void localMatrixMultiplication_OMP(int* A_local, int* B, int* C_local, int N, int local_M) {
    // OpenMP directive: the threads within this MPI process divide the 'i' loop (rows of the local chunk).
    #pragma omp parallel for
    for (int i = 0; i < local_M; i++) {
        for (int j = 0; j < N; j++) {
            long long sum = 0;
            for (int k = 0; k < N; k++) {
                sum += (long long)A_local[i * N + k] * B[k * N + j];
            }
            C_local[i * N + j] = (int)sum;
        }
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI and request thread support (MPI_THREAD_FUNNELED is the correct standard)
    int provided;
    // FIX: Changed MPI_THREAD_FUNNELS to MPI_THREAD_FUNNELED
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 512; 
    
    if (rank == 0 && argc > 1) {
        try { N = stoi(argv[1]); } catch (...) { N = 512; }
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "--- Hybrid MPI + OpenMP ---" << endl;
        cout << "MPI Processes: " << size << ", OMP Threads per Process: " << omp_get_max_threads() << endl;
    }

    // Allocation and distribution setup (Same as MPI-Only)
    int total_elements = N * N;
    int* A = nullptr;
    int* B = nullptr;
    int* C = nullptr;
    int* sendcounts = nullptr;
    int* displs = nullptr;
    
    if (rank == 0) {
        A = allocateMatrix(total_elements);
        B = allocateMatrix(total_elements);
        C = allocateMatrix(total_elements);
        initializeMatrix(A, N);
        initializeMatrix(B, N);
        sendcounts = new int[size];
        displs = new int[size];
        int sum = 0;
        for (int i = 0; i < size; i++) {
            int rows = N / size;
            if (i < N % size) rows++;
            sendcounts[i] = rows * N; 
            displs[i] = sum;          
            sum += sendcounts[i];
        }
    }

    int local_elements = 0;
    if (rank == 0) local_elements = (size > 0 && sendcounts) ? sendcounts[0] : 0; 
    else {
        int rows_for_this_rank = N / size;
        if (rank < N % size) rows_for_this_rank++;
        local_elements = rows_for_this_rank * N;
    }
    
    int local_M = local_elements / N; 
    int* A_local = allocateMatrix(local_elements);
    int* C_local = allocateMatrix(local_elements);
    if (rank != 0) B = allocateMatrix(total_elements); 

    MPI_Bcast(B, total_elements, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    double start_time = 0;
    if (rank == 0) start_time = MPI_Wtime();
    
    MPI_Scatterv(A, sendcounts, displs, MPI_INT, 
                 A_local, local_elements, MPI_INT, 
                 0, MPI_COMM_WORLD);

    // HYBRID COMPUTATION: OpenMP is called here to parallelize the local work
    if (local_elements > 0) {
       localMatrixMultiplication_OMP(A_local, B, C_local, N, local_M);
    }
    
    MPI_Gatherv(C_local, local_elements, MPI_INT, 
                C, sendcounts, displs, MPI_INT, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        double end_time = MPI_Wtime();
        double duration = (end_time - start_time) * 1000.0; 
        cout << "MPI + OpenMP Multiplication Time: " << duration << " ms" << endl;
    }

    // Memory Cleanup
    if (A) deallocateMatrix(A);
    if (B) deallocateMatrix(B);
    if (C) deallocateMatrix(C);
    if (A_local) deallocateMatrix(A_local);
    if (C_local) deallocateMatrix(C_local);
    if (sendcounts) delete[] sendcounts;
    if (displs) delete[] displs;
    
    MPI_Finalize();
    return 0;
}