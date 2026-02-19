#include <iostream>
#include <cstdlib>
#include <chrono>
#include <string>
#include <mpi.h>

using namespace std;

// Function: allocateMatrix
//  Allocates memory for a 1D array of a specified size. Used for A, B, C, and their local chunks.
//  size (int) - The number of elements to allocate.
// Returns: int* - Pointer to the allocated memory.
int* allocateMatrix(int size) {
    int* matrix = new int[size]; 
    return matrix;
}

// Function: deallocateMatrix
// Frees the memory allocated for a matrix.
// matrix (int*) - Pointer to the matrix memory.
void deallocateMatrix(int* matrix) {
    delete[] matrix;
}

// Function: initializeMatrix
// Description: Fills an N*N matrix with random integer values (0-99). Only run on the root (Rank 0).
// Parameters: matrix (int*) - Pointer to the matrix memory, size (int) - The dimension N (matrix is N x N).
void initializeMatrix(int* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 100;
    }
}

// Function: localMatrixMultiplication
//  Computes the local chunk of C (C_local = A_local * B) sequentially on the CPU.
//  A_local (int*) - Local rows of A, B (int*) - Full B matrix, C_local (int*) - Local result C rows, 
//             N (int) - Matrix dimension, local_M (int) - Number of rows in A_local/C_local.
void localMatrixMultiplication(int* A_local, int* B, int* C_local, int N, int local_M) {
    // Standard matrix multiplication loop structure
    for (int i = 0; i < local_M; i++) {
        for (int j = 0; j < N; j++) {
            long long sum = 0;
            for (int k = 0; k < N; k++) {
                // Calculation: A[i][k] * B[k][j]
                sum += (long long)A_local[i * N + k] * B[k * N + j];
            }
            C_local[i * N + j] = (int)sum;
        }
    }
}

int main(int argc, char* argv[]) {
    
    // MPI Initialization and Rank/Size retrieval
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 512; // Default matrix size (NxN)
    
    // Broadcast N to all processes
    if (rank == 0 && argc > 1) {
        try { N = stoi(argv[1]); } catch (...) { N = 512; }
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (size < 2) {
        if (rank == 0) cerr << "ERROR: MPI requires at least 2 processes." << endl;
        MPI_Finalize();
        return 1;
    }

    int total_elements = N * N;
    int* A = nullptr;
    int* B = nullptr;
    int* C = nullptr;
    int* sendcounts = nullptr;
    int* displs = nullptr;
    
    if (rank == 0) {
        // Root initializes global matrices
        A = allocateMatrix(total_elements);
        B = allocateMatrix(total_elements);
        C = allocateMatrix(total_elements);
        initializeMatrix(A, N);
        initializeMatrix(B, N);
        
        // Calculate distribution parameters (sendcounts and displacements)
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

    // Determine the size of the local chunk for this rank
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
    
    // Non-root processes allocate space for the entire B matrix
    if (rank != 0) B = allocateMatrix(total_elements); 

    // Communication Step 1: Broadcast the entire B matrix to all ranks
    MPI_Bcast(B, total_elements, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    double start_time = 0;
    if (rank == 0) start_time = MPI_Wtime();
    
    // Communication Step 2: Scatter the rows of A to all ranks
    MPI_Scatterv(A, sendcounts, displs, MPI_INT, 
                 A_local, local_elements, MPI_INT, 
                 0, MPI_COMM_WORLD);

    // COMPUTATION
    if (local_elements > 0) {
        localMatrixMultiplication(A_local, B, C_local, N, local_M); 
    }
    
    // Communication Step 3: Gather the computed C_local chunks back to rank 0
    MPI_Gatherv(C_local, local_elements, MPI_INT, 
                C, sendcounts, displs, MPI_INT, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        double end_time = MPI_Wtime();
        double duration = (end_time - start_time) * 1000.0; 
        cout << "MPI-Only Multiplication Time: " << duration << " ms" << endl;
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
