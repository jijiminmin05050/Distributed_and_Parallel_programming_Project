#include <iostream>
#include <vector>
#include <algorithm> // For std::sort and std::merge
#include <cmath>     // For std::pow
#include <chrono>    // For timing
#include <cstdlib>   // For std::srand and std::rand
#include "mpi.h"     // Required for MPI functions

// Purpose: Generates a large vector of random integers.
// Execution: Sequential (runs on the root process).
void generate_random_vector(std::vector<int>& vec, int size) {
    vec.resize(size);
    std::srand(std::time(0) + MPI::COMM_WORLD.Get_rank());
    for (int& val : vec) {
        val = std::rand() % 1000000; // Random numbers up to 1 million
    }
}

// Purpose: Merges sorted chunks from all processes into a single sorted array
//          on the root process (Rank 0).
// Execution: Sequential (runs on the root process).
void final_merge(std::vector<int>& final_array, const std::vector<std::vector<int>>& sorted_chunks) {
    if (sorted_chunks.empty()) return;

    // Start with the first chunk
    final_array = sorted_chunks[0];

    // Iteratively merge the current result with the next chunk
    for (size_t i = 1; i < sorted_chunks.size(); ++i) {
        std::vector<int> temp_result;
        temp_result.reserve(final_array.size() + sorted_chunks[i].size());

        // Use std::merge to efficiently combine two sorted ranges
        std::merge(
            final_array.begin(), final_array.end(),
            sorted_chunks[i].begin(), sorted_chunks[i].end(),
            std::back_inserter(temp_result)
        );
        final_array = std::move(temp_result);
    }
}

int main(int argc, char* argv[]) {
    // 1. MPI Initialization
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    if (argc < 2) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <array_size>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Array size and chunk calculation
    const int N = std::atoi(argv[1]);
    const int chunk_size = N / size;
    const int remainder = N % size;

    std::vector<int> global_array;
    std::vector<int> local_chunk(chunk_size + (rank < remainder ? 1 : 0));
    int local_n = local_chunk.size();

    // 2. Data Generation and Distribution (Rank 0 only)
    if (rank == 0) {
        generate_random_vector(global_array, N);
        std::cout << "--- MPI Parallel Quicksort Baseline ---\n";
        std::cout << "Array Size: " << N << " | Processes: " << size << "\n";
    }

    // Array of 'sendcounts' and 'displs' needed for MPI_Gatherv later
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    int current_displ = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = chunk_size + (i < remainder ? 1 : 0);
        displs[i] = current_displ;
        current_displ += sendcounts[i];
    }
    
    // Use MPI_Scatterv for distribution (required when chunk sizes are unequal)
    auto start_time = std::chrono::high_resolution_clock::now();
    MPI_Scatterv(
        global_array.data(), // Send buffer (only significant at rank 0)
        sendcounts.data(),   // Count of elements sent to each process
        displs.data(),       // Displacement (offset) for each send
        MPI_INT,             // Data type
        local_chunk.data(),  // Receive buffer
        local_n,             // Count of received elements
        MPI_INT,
        0,                   // Root process (Rank 0)
        MPI_COMM_WORLD
    );

    // 3. Local Sort (Conquer)
    // Purpose: Sorts the local chunk using the optimized C++ standard library sort (Quicksort variant).
    std::sort(local_chunk.begin(), local_chunk.end());

    // 4. Gather Results (Rank 0 only)
    std::vector<std::vector<int>> all_sorted_chunks(size);
    std::vector<int> all_received_data;

    if (rank == 0) {
        // Allocate space for all received data
        all_received_data.resize(N);
    }
    
    // Gather all sorted chunks back to the root process
    MPI_Gatherv(
        local_chunk.data(),
        local_n,
        MPI_INT,
        all_received_data.data(), // Receive buffer (only significant at rank 0)
        sendcounts.data(),        // Count of elements received from each process
        displs.data(),            // Displacement (offset) for each receive
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );
    
    // 5. Final Merge (Rank 0 only)
    if (rank == 0) {
        // Reconstruct the array of vectors for the merge function
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

        // Optional: Verification check
        // if (std::is_sorted(final_sorted_array.begin(), final_sorted_array.end())) {
        //     std::cout << "Verification: Array is correctly sorted.\n";
        // } else {
        //     std::cout << "Verification: ERROR - Array is NOT sorted.\n";
        // }
    }

    MPI_Finalize();
    return 0;
}
