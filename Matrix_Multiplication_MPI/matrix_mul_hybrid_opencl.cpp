#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

// Try different OpenCL header locations
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

// Function to read the kernel source from a file
std::string read_kernel_source(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        // Print error to both console streams for visibility
        std::cerr << "Error: Could not open kernel file: " << filename << std::endl;
        std::cout << "Error: Could not open kernel file: " << filename << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Allocates memory for a 1D array of a specified size.
int* allocateMatrix(int size) {
    int* matrix = new int[size]; 
    return matrix;
}

// Frees the memory allocated for a matrix.
void deallocateMatrix(int* matrix) {
    delete[] matrix;
}

// Fills an N*N matrix with random integer values (0-99).
void initializeMatrix(int* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 100;
    }
}

// Function: localMatrixMultiplication_OCL
// Description: Offloads the C_local = A_local * B computation to the OpenCL device (GPU).
// Parameters: A_local (int*), B (int*), C_local (int*) - Host buffers, N (int), local_M (int), 
//             context/queue/kernel - OpenCL objects for device communication.
void localMatrixMultiplication_OCL(int* A_local, int* B, int* C_local, int N, int local_M, cl_context context, cl_command_queue queue, cl_kernel kernel) {
    
    cl_int err;
    int local_elements = local_M * N;

    // 1. Create Buffers on Device (Allocate GPU memory and copy Host data A_local and B)
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, local_elements * sizeof(int), A_local, &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * N * sizeof(int), B, &err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, local_elements * sizeof(int), NULL, &err);

    // 2. Set Kernel Arguments (Map the device buffers and scalar values to the kernel function)
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &N);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &local_M);

    // 3. Define Work Group Size (Set up parallel execution dimensions)
    size_t global_work_size[2] = {(size_t)local_M, (size_t)N};
    size_t local_work_size[2] = {16, 16}; 
    
    // Safety check and adjustment for optimal grid size
    if (local_M < 16) local_work_size[0] = local_M;
    if (N < 16) local_work_size[1] = N;

    // Ensure Global Work Size is a multiple of Local Work Size
    global_work_size[0] = (size_t)ceil((float)local_M / local_work_size[0]) * local_work_size[0];
    global_work_size[1] = (size_t)ceil((float)N / local_work_size[1]) * local_work_size[1];


    // 4. Enqueue Kernel (Send command to execute the matrix_multiply_kernel on the device)
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    // 5. Read Result Back to Host Memory (Block and wait for the GPU to finish, then transfer C_local back)
    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, local_elements * sizeof(int), C_local, 0, NULL, NULL);

    // 6. Cleanup (Release device memory resources)
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
}

// --- Main Hybrid Logic (OpenCL Setup and MPI distribution) ---

int main(int argc, char* argv[]) {
    
    // MPI Setup
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 512; 
    
    if (rank == 0 && argc > 1) {
        try { N = stoi(argv[1]); } catch (...) { N = 512; }
    }
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // OpenCL Device Setup 
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_uint num_devices;
    cl_uint num_platforms;
    cl_int err;

    // Report the required compilation command if it's Rank 0 and compilation failed previously
    if (rank == 0) {
        cout << "--- Hybrid MPI + OpenCL ---" << endl;
        cout << "If 'CL/cl.h' error persists, use this explicit compilation command:" << endl;
        cout << "mpicxx matrix_mul_hybrid_opencl.cpp -o hybrid_ocl -I/System/Library/Frameworks/OpenCL.framework/Versions/Current/Headers/ -framework OpenCL" << endl;
    }

    err = clGetPlatformIDs(1, &platform_id, &num_platforms);
    // Try to get a GPU first, then fallback to CPU
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
    if (err != CL_SUCCESS) { err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &num_devices); }
    
    if (err != CL_SUCCESS) {
        if (rank == 0) cerr << "Error: No suitable OpenCL device found (GPU or CPU)." << endl;
        MPI_Finalize();
        return 1;
    }
    
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);

    // Load kernel source from the external file
    string kernel_source_str = read_kernel_source("matrix_multiply.cl");
    if (kernel_source_str.empty()) {
        MPI_Finalize();
        return 1;
    }
    const char *source_str = kernel_source_str.c_str();
    
    program = clCreateProgramWithSource(context, 1, &source_str, NULL, &err);
    
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    
    // Check for build errors and print the log if needed
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
        cerr << "OpenCL Build Error on Rank " << rank << ":\n" << build_log.data() << endl;
    }

    kernel = clCreateKernel(program, "matrix_multiply_kernel", &err);

    // Matrix and MPI setup (identical to other MPI files)
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

    // HYBRID COMPUTATION
    if (local_elements > 0) {
        localMatrixMultiplication_OCL(A_local, B, C_local, N, local_M, context, queue, kernel);
    }
    
    MPI_Gatherv(C_local, local_elements, MPI_INT, 
                C, sendcounts, displs, MPI_INT, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        double end_time = MPI_Wtime();
        double duration = (end_time - start_time) * 1000.0; 
        cout << "MPI + OpenCL Multiplication Time: " << duration << " ms" << endl;
    }

    // CLEANUP
    if (A) deallocateMatrix(A);
    if (B) deallocateMatrix(B);
    if (C) deallocateMatrix(C);
    if (A_local) deallocateMatrix(A_local);
    if (C_local) deallocateMatrix(C_local);
    if (sendcounts) delete[] sendcounts;
    if (displs) delete[] displs;
    
    // OpenCL Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    MPI_Finalize();
    return 0;
}
