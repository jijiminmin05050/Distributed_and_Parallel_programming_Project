#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <mpi.h>
#include <omp.h> // Required for OpenMP integration
#include <cstddef> // Required for offsetof

// --- 1. Data Structures ---

// Structure used by the Master (Rank 0) to parse the data file.
// TrafficData represents a single observation (light_id, car_count in one time slice).
struct TrafficData {
    int light_id;
    int car_count;
};

// Structure for local aggregation inside 'process_local_data'
struct LocalMetrics {
    long long total_cars = 0;
    int observation_count = 0; // Number of times this light was measured
    int max_spike = 0;         // The highest single car count recorded
};

// Structure used for communication: aggregating results from slaves to master.
// NEW FIELDS added for advanced analysis: observation_count and max_spike.
struct AggregatedData {
    int light_id;
    long long total_cars;
    int observation_count; 
    int max_spike;         
};

// --- 2. MPI Custom Type Creation ---

// Function to create an MPI derived datatype for the AggregatedData struct
MPI_Datatype create_mpi_aggregated_data() {
    MPI_Datatype type;
    // Four fields: int, long long, int, int
    int blocklengths[4] = {1, 1, 1, 1}; 
    
    MPI_Aint offsets[4];
    
    // Use offsetof macro to correctly calculate offsets within the struct.
    offsets[0] = offsetof(AggregatedData, light_id);
    offsets[1] = offsetof(AggregatedData, total_cars);
    offsets[2] = offsetof(AggregatedData, observation_count);
    offsets[3] = offsetof(AggregatedData, max_spike);
    
    MPI_Datatype types[4] = {MPI_INT, MPI_LONG_LONG, MPI_INT, MPI_INT}; 
    
    MPI_Type_create_struct(4, blocklengths, offsets, types, &type);
    MPI_Type_commit(&type);
    
    return type;
}

// --- 3. Local Processing Function (OpenMP Performance Fix) ---

/**
 * Purpose: Processes a chunk of data (TrafficData records) to create a local congestion map.
 * This function uses OpenMP with Thread-Local Storage (TLS) to efficiently calculate
 * Total Cars, Observation Count, and Max Spike.
 */
std::map<int, LocalMetrics> process_local_data(const std::vector<TrafficData>& local_data) {
    // Final map to store the aggregated results
    std::map<int, LocalMetrics> local_congestion_map;
    
    // Get the maximum number of threads available
    int num_threads = 1;
    #ifdef _OPENMP
        num_threads = omp_get_max_threads();
    #endif

    // Vector to hold a separate map for each thread (Thread-Local Storage strategy)
    std::vector<std::map<int, LocalMetrics>> thread_maps(num_threads);

    #pragma omp parallel default(none) shared(thread_maps, local_data, num_threads)
    {
        // Get thread ID
        int tid = 0;
        #ifdef _OPENMP
            tid = omp_get_thread_num();
        #endif

        // Parallel loop: each thread updates its own private map
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < local_data.size(); ++i) {
            int light_id = local_data[i].light_id;
            int car_count = local_data[i].car_count;

            // Update thread-specific metrics
            thread_maps[tid][light_id].total_cars += car_count;
            thread_maps[tid][light_id].observation_count += 1; 
            
            // Track max spike
            if (car_count > thread_maps[tid][light_id].max_spike) {
                thread_maps[tid][light_id].max_spike = car_count;
            }
        }
    } // End of parallel region

    // Sequential merge: Merge all thread-local maps into the final map
    for (const auto& t_map : thread_maps) {
        for (const auto& pair : t_map) {
            local_congestion_map[pair.first].total_cars += pair.second.total_cars;
            local_congestion_map[pair.first].observation_count += pair.second.observation_count;
            
            // Merge max_spike: take the maximum observed value
            if (pair.second.max_spike > local_congestion_map[pair.first].max_spike) {
                 local_congestion_map[pair.first].max_spike = pair.second.max_spike;
            }
        }
    }

    return local_congestion_map;
}

// --- 4. Master (Rank 0) File Reading and Distribution ---

// Master function to read the entire file and coordinate data scattering
std::vector<TrafficData> master_read_and_distribute(const std::string& filename, int world_size, 
                                                   std::vector<int>& sendcounts, 
                                                   std::vector<int>& displs) {
    std::ifstream file(filename);
    std::vector<TrafficData> all_records;
    std::string line;
    int total_records = 0; // Initialize for Bcast

    if (!file.is_open()) {
        std::cerr << "Master Error: Could not open data file: " << filename << std::endl;
        // Bcast 0 to signal error to slaves before returning
        MPI_Bcast(&total_records, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return all_records;
    }
    
    // Skip the header line
    if (std::getline(file, line)) {}
    
    // 1. Read all data records
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string ts_str, id_str, count_str;
        
        // Data format: timestamp,light_id,car_count
        if (std::getline(ss, ts_str, ',') &&
            std::getline(ss, id_str, ',') &&
            std::getline(ss, count_str, ',')) {

            try {
                // We only store the light ID and car count for distribution
                all_records.push_back({std::stoi(id_str), std::stoi(count_str)});
            } catch (const std::exception& e) {
                std::cerr << "Master Error: Invalid data format in line: " << line << std::endl;
                continue;
            }
        }
    }

    // 2. Determine send counts and displacements for distribution
    total_records = all_records.size();
    
    // ROBUSTNESS FIX: Broadcast the total record count to all slaves *after* reading
    MPI_Bcast(&total_records, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int base_count = total_records / world_size;
    int remainder = total_records % world_size;

    sendcounts.resize(world_size);
    displs.resize(world_size);
    displs[0] = 0;

    for (int i = 0; i < world_size; ++i) {
        // Correct logic for Scatterv
        sendcounts[i] = base_count + (i < remainder ? 1 : 0);
        if (i > 0) {
            displs[i] = displs[i-1] + sendcounts[i-1];
        }
    }

    std::cout << "Master: Read " << total_records << " total records. Distributing among " << world_size << " processes." << std::endl;

    return all_records;
}

// --- 5. Final Analysis and Report Generation (Master Only) ---

// Purpose: Aggregates the final global map and prints the top N.
void analyze_congestion(const std::vector<AggregatedData>& global_results, int top_n, double total_time, double processing_time, int world_size) {
    
    // Map to hold FINAL aggregated metrics (used for calculating ACI)
    struct FinalMetrics {
        long long total_cars = 0;
        long long observation_count = 0;
        int max_spike = 0;
        double ACI = 0.0; // Average Congestion Index
    };
    std::map<int, FinalMetrics> final_metrics_map;

    // 1. Global Aggregation: Sum up all metrics from all processes
    for (const auto& data : global_results) {
        final_metrics_map[data.light_id].total_cars += data.total_cars;
        // observation_count needs to be summed globally
        final_metrics_map[data.light_id].observation_count += data.observation_count;
        
        // max_spike takes the maximum observed value across all processes
        if (data.max_spike > final_metrics_map[data.light_id].max_spike) {
            final_metrics_map[data.light_id].max_spike = data.max_spike;
        }
    }

    // 2. Calculate ACI and convert map to vector of pairs for sorting
    // We will sort by ACI (Average Congestion Index)
    std::vector<std::pair<double, int>> sorted_lights_by_aci;
    for (auto& pair : final_metrics_map) {
        if (pair.second.observation_count > 0) {
            // ACI = Total Cars / Total Observations (Average intensity)
            pair.second.ACI = (double)pair.second.total_cars / pair.second.observation_count;
        } else {
            pair.second.ACI = 0.0;
        }
        sorted_lights_by_aci.push_back({pair.second.ACI, pair.first});
    }

    if (sorted_lights_by_aci.empty()) {
        std::cout << "\nNo traffic data was processed." << std::endl;
        return;
    }

    // 3. Sort the vector (descending by ACI)
    std::sort(sorted_lights_by_aci.rbegin(), sorted_lights_by_aci.rend());

    // 4. Print the top N (using the new ACI and Max Spike metrics)
    std::cout << "\n=========================================================================" << std::endl;
    std::cout << "          Top " << top_n << " Traffic Lights by Average Congestion Index (ACI)" << std::endl;
    std::cout << "=========================================================================" << std::endl;
    std::cout << std::left << std::setw(15) << "Rank" 
              << std::setw(20) << "Traffic Light ID" 
              << std::setw(20) << "ACI (Avg Cars)" 
              << std::setw(20) << "Max Car Spike" << std::endl;
    std::cout << "-------------------------------------------------------------------------" << std::endl;

    int rank = 1;
    for (const auto& pair : sorted_lights_by_aci) {
        if (rank > top_n) break;
        int light_id = pair.second;
        // Use .at() for safe access since the key must exist from the sorting vector
        const auto& metrics = final_metrics_map.at(light_id);
        
        std::cout << std::left << std::setw(15) << rank++
                  << std::setw(20) << light_id
                  << std::setw(20) << std::fixed << std::setprecision(2) << metrics.ACI
                  << std::setw(20) << metrics.max_spike << std::endl;
    }
    std::cout << "=========================================================================" << std::endl;

    // 5. Print Performance Metrics
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n--- Performance Summary (P=" << world_size << ") ---" << std::endl;
    std::cout << "Total Wall Clock Time: " << total_time << " seconds" << std::endl;
    std::cout << "Local Processing Time (per process estimate): " << processing_time << " seconds" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
}

// --- 6. Main MPI Execution ---

int main(int argc, char* argv[]) {
    // We expect 2 arguments after the executable name: data_file and Top N
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <data_file> <top_N>" << std::endl;
        std::cerr << "Example: mpirun -np 4 " << argv[0] << " traffic_data.txt 5" << std::endl;
        return 1;
    }

    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const std::string data_file = argv[1];
    const int top_n = std::atoi(argv[2]);

    double total_start_time = MPI_Wtime();
    double processing_start_time = 0.0;
    double processing_end_time = 0.0;
    int total_records = 0; 

    // Create custom MPI type for AggregatedData (used for result collection)
    MPI_Datatype MPI_AGGREGATED_DATA = create_mpi_aggregated_data();

    // --- Master (Rank 0) Logic ---
    if (world_rank == 0) {
        std::vector<int> sendcounts;
        std::vector<int> displs;
        
        // Master reads file and Bcasts total_records internally
        std::vector<TrafficData> all_records = master_read_and_distribute(data_file, world_size, sendcounts, displs);

        // Update total_records for the master's local context
        total_records = all_records.size();

        if (total_records == 0) {
            // Master failed to read and Bcast 0. Abort gracefully.
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // --- Data Scatter ---

        // Local buffer for the master's chunk
        std::vector<TrafficData> master_local_data(sendcounts[0]); 
        
        // Scatter the data records to all processes. TrafficData is just two MPI_INTs.
        MPI_Scatterv(all_records.data(),             // Full send buffer (master only)
                     sendcounts.data(),              // Send count for each process
                     displs.data(),                  // Displacement for each process
                     MPI_2INT,                       // Type of element being sent (2 integers: light_id, car_count)
                     master_local_data.data(),       // Receive buffer (master only)
                     sendcounts[0],                  // Receive count
                     MPI_2INT,                       // Receive type
                     0,                              // Root process
                     MPI_COMM_WORLD);
        
        // --- Local Processing (Timing Start) ---
        processing_start_time = MPI_Wtime();
        // Master process runs local processing, returns map of LocalMetrics
        std::map<int, LocalMetrics> master_map = process_local_data(master_local_data);
        processing_end_time = MPI_Wtime();
        // --- Local Processing (Timing End) ---
        
        // --- Result Collection (Gatherv) ---

        // 1. Convert master's LocalMetrics map to the AggregatedData format for MPI
        std::vector<AggregatedData> master_results;
        for (const auto& pair : master_map) {
            master_results.push_back({
                pair.first, 
                pair.second.total_cars,
                pair.second.observation_count,
                pair.second.max_spike
            });
        }

        // 2. Prepare master send/receive counts for Gatherv
        int *recvcounts = new int[world_size];
        int *rdispls = new int[world_size];

        // Collect all result sizes (efficiently using blocking MPI_Gather)
        int master_size = master_results.size();
        MPI_Gather(&master_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Calculate displacements for Gatherv
        rdispls[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            rdispls[i] = rdispls[i-1] + recvcounts[i-1];
        }

        int total_results_count = rdispls[world_size - 1] + recvcounts[world_size - 1];
        std::vector<AggregatedData> global_results(total_results_count);

        // Gatherv master's results (master is root)
        MPI_Gatherv(master_results.data(), 
                    master_results.size(), 
                    MPI_AGGREGATED_DATA,
                    global_results.data(), 
                    recvcounts, 
                    rdispls, 
                    MPI_AGGREGATED_DATA,
                    0, 
                    MPI_COMM_WORLD);
        
        double total_end_time = MPI_Wtime();
        double total_time = total_end_time - total_start_time;
        double processing_time = processing_end_time - processing_start_time; // Using master's processing time as estimate
        
        // 3. Final Analysis and Report
        analyze_congestion(global_results, top_n, total_time, processing_time, world_size);
        
        delete[] recvcounts;
        delete[] rdispls;

    } 
    // --- Slave (Rank > 0) Logic ---
    else {
        // Slaves receive the actual total record count from the master
        MPI_Bcast(&total_records, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (total_records == 0) {
            // Master failed to read data and broadcasted 0 before aborting.
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Calculation of expected receive size based on the actual total records
        int base_count = total_records / world_size;
        int remainder = total_records % world_size;
        
        int local_receive_count = base_count + (world_rank < remainder ? 1 : 0);

        std::vector<TrafficData> slave_local_data(local_receive_count);

        // Slave Receive
        MPI_Scatterv(nullptr,                     // Null send buffer
                     nullptr,                     // Null send counts
                     nullptr,                     // Null displacements
                     MPI_2INT,                    // Type of element being received (2 integers)
                     slave_local_data.data(),     // Receive buffer
                     local_receive_count,         // Receive count (now correctly calculated)
                     MPI_2INT,                    // Receive type
                     0,                           // Root process
                     MPI_COMM_WORLD);

        // --- Local Processing (Timing Start) ---
        processing_start_time = MPI_Wtime();
        // Slave process runs local processing, returns map of LocalMetrics
        std::map<int, LocalMetrics> slave_map = process_local_data(slave_local_data);
        processing_end_time = MPI_Wtime();
        // --- Local Processing (Timing End) ---
        
        // --- Result Serialization ---
        // Convert local map to the AggregatedData format for MPI
        std::vector<AggregatedData> slave_results;
        for (const auto& pair : slave_map) {
            slave_results.push_back({
                pair.first, 
                pair.second.total_cars,
                pair.second.observation_count,
                pair.second.max_spike
            });
        }
        
        // --- Result Collection (Slave Send) ---
        int slave_result_size = slave_results.size();
        
        // 1. Slaves send their result size to master 
        int *recvcounts = new int[world_size];
        MPI_Gather(&slave_result_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
        delete[] recvcounts; // Slaves don't need this array after the call
        
        // Gatherv slave's results (slave sends)
        // Slaves only need the send arguments, so the receive arguments are null.
        MPI_Gatherv(slave_results.data(), 
                    slave_results.size(), 
                    MPI_AGGREGATED_DATA,
                    nullptr,        // Null receive buffer on slave
                    nullptr,        // Null recvcounts on slave
                    nullptr,        // Null rdispls on slave
                    MPI_AGGREGATED_DATA,
                    0, 
                    MPI_COMM_WORLD);
    }

    MPI_Type_free(&MPI_AGGREGATED_DATA);
    MPI_Finalize();

    return 0;
}
