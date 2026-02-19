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
// --- OpenCL Utility Functions (Host Code) ---

// Purpose: Reads the OpenCL kernel source code from a file into a string.
// Execution: Sequential (runs on the root process for setup).
std::string read_kernel_file(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Failed to open kernel file: ") + filename);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Purpose: Merges sorted chunks from all processes into a single sorted array
//          on the root process (Rank 0).
// Execution: Sequential (runs on the root process).
void final_merge(std::vector<int>& final_array, const std::vector<std::vector<int>>& sorted_chunks) {
    if (sorted_chunks.empty()) return;

    final_array = sorted_chunks[0];
    for (size_t i = 1; i < sorted_chunks.size(); ++i) {
        std::vector<int> temp_result;
        temp_result.reserve(final_array.size() + sorted_chunks[i].size());

        std::merge(
            final_array.begin(), final_array.end(),
            sorted_chunks[i].begin(), sorted_chunks[i].end(),
            std::back_inserter(temp_result)
        );
        final_array = std::move(temp_result);
    }
}

// Purpose: Generates a large vector of random integers.
// Execution: Sequential (runs on the root process).
void generate_random_vector(std::vector<int>& vec, int size) {
    vec.resize(size);
    std::srand(std::time(0) + MPI::COMM_WORLD.Get_rank());
    for (int& val : vec) {
        val = std::rand() % 1000000; // Random numbers up to 1 million
    }
}

// --- Main Program ---

int main(int argc, char* argv[]) {
    // 1. MPI Initialization
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <array_size>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Array size and chunk calculation
    const int N_total = std::atoi(argv[1]);
    int local_n = 0;
    
    // Determine the size of the local chunk for each process
    const int chunk_size = N_total / size;
    const int remainder = N_total % size;
    local_n = chunk_size + (rank < remainder ? 1 : 0);

    // Find the next power of 2 for Bitonic Sort (OpenCL requirement)
    int n_ceil = 1;
    while (n_ceil < local_n) {
        n_ceil <<= 1;
    }
    const int local_n_padded = n_ceil;

    std::vector<int> global_array;
    std::vector<int> local_chunk(local_n_padded); // Padded for OpenCL
    
    // Data Generation and Distribution Setup
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    int current_displ = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = chunk_size + (i < remainder ? 1 : 0);
        displs[i] = current_displ;
        current_displ += sendcounts[i];
    }
    
    if (rank == 0) {
        generate_random_vector(global_array, N_total);
        std::cout << "--- Hybrid MPI + OpenCL Quicksort ---\n";
        std::cout << "Array Size: " << N_total << " | Processes: " << size << "\n";
    }

    // Scatter the non-padded data
    auto start_time = std::chrono::high_resolution_clock::now();
    MPI_Scatterv(
        global_array.data(), sendcounts.data(), displs.data(), MPI_INT,
        local_chunk.data(), local_n, MPI_INT,
        0, MPI_COMM_WORLD
    );

    // Pad the local chunk with sentinel values (e.g., INT_MAX) for Bitonic Sort
    for (int i = local_n; i < local_n_padded; ++i) {
        local_chunk[i] = 2000000; // Placeholder value larger than max random
    }

    // --- 3. Local Sort using OpenCL (Conquer) ---
    // (Error handling is simplified for brevity)
    
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
    
    // Read kernel source
    std::string kernel_source_str;
    try {
        kernel_source_str = read_kernel_file("bitonic_sort.cl");
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << ": Kernel read error: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }
    const char* kernel_source = kernel_source_str.c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
    
    // Compile OpenCL program
    if (clBuildProgram(program, 1, &device, NULL, NULL, NULL) != CL_SUCCESS) {
        // Output build log on failure
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
        std::cerr << "OpenCL Build Error (Rank " << rank << "):\n" << build_log.data() << std::endl;
        MPI_Finalize();
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "bitonic_sort", NULL);

    // Create device buffer
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                   sizeof(int) * local_n_padded, local_chunk.data(), NULL);
    
    // Execute Bitonic Sort
    size_t global_size = local_n_padded / 2;
    size_t local_size = 256; // Standard local work group size
    if (global_size < local_size) local_size = global_size;
    
    // Outer loop for merge stage size (m)
    for (int m = 2; m <= local_n_padded; m *= 2) {
        // Inner loop for compare-exchange step size (j)
        for (int j = m / 2; j > 0; j /= 2) {
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
            clSetKernelArg(kernel, 1, sizeof(cl_int), &m);
            clSetKernelArg(kernel, 2, sizeof(cl_int), &j);
            clSetKernelArg(kernel, 3, sizeof(cl_int), &local_n_padded);

            clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
        }
    }
    clFinish(queue);

    // Read result back to host
    clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, sizeof(int) * local_n_padded, local_chunk.data(), 0, NULL, NULL);

    // Clean up OpenCL resources
    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    // --- 4. Gather Results (Rank 0 only) ---
    // Remove padded elements (they were at the end)
    local_chunk.resize(local_n);

    std::vector<int> all_received_data;
    if (rank == 0) {
        all_received_data.resize(N_total);
    }
    
    MPI_Gatherv(
        local_chunk.data(), local_n, MPI_INT,
        all_received_data.data(), sendcounts.data(), displs.data(), MPI_INT,
        0, MPI_COMM_WORLD
    );
    
    // 5. Final Merge (Rank 0 only)
    if (rank == 0) {
        std::vector<std::vector<int>> all_sorted_chunks(size);
        for (int i = 0; i < size; ++i) {
            all_sorted_chunks[i].assign(
                all_received_data.begin() + displs[i],
                all_received_data.begin() + displs[i] + sendcounts[i]
            );
        }

        std::vector<int> final_sorted_array;
        final_merge(final_sorted_array, all_sorted_chunks);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Execution Time: " << duration.count() << " ms\n";
    }

    MPI_Finalize();
    return 0;
}
