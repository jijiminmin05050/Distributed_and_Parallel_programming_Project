// OpenCL Kernel File: matrix_multiply.cl
// This kernel is responsible for performing the core matrix multiplication 
// for a small block of rows assigned to a specific MPI process.

// Purpose: Computes C[i][j] = sum(A[i][k] * B[k][j]) for the local row chunk.
// Execution: This code is compiled and run on the OpenCL device (typically a GPU or specialized CPU).
__kernel void matrix_multiply_kernel(
    // Input/Output Buffers (Global memory access)
    __global const int* A,  // Local row block of Matrix A (M x N)
    __global const int* B,  // Full Matrix B (N x N)
    __global int* C,        // Result Matrix C block (M x N)
    
    // Scalar Inputs
    int N,                  // The dimension of the square matrices (N x N)
    int M                   // The number of rows in the local A and C chunk assigned to this MPI process
) {
    // get_global_id(0) returns the current work-item's index in the first dimension (row index 'i').
    int i = get_global_id(0); 
    
    // get_global_id(1) returns the current work-item's index in the second dimension (column index 'j').
    int j = get_global_id(1); 

    // Safety check: Ensure the work-item is within the bounds of the local matrix block (M x N).
    if (i < M && j < N) {
        long sum = 0;
        
        // The dot product loop (k-loop) for computing the single element C[i][j].
        for (int k = 0; k < N; k++) {
            // A[i * N + k]: Accesses element in row 'i' of A.
            // B[k * N + j]: Accesses element in row 'k', column 'j' of B (B is stored row-major).
            sum += (long)A[i * N + k] * B[k * N + j];
        }
        
        C[i * N + j] = (int)sum;
    }
}