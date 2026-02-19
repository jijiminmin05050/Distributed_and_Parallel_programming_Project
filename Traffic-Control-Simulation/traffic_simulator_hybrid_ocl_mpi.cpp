#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <iomanip>
#include <string>
#include <cmath>
#include <stdexcept>
#include <mpi.h>

// --- Custom Data Structures ---

/**
 * Structure to hold a single traffic record read from the file.
 * The Master scatters a vector of these to the Slaves.
 * The timestamp is converted to total seconds from 00:00 for calculation.
 */
struct TrafficData {
    int timestamp;  // Total seconds from 00:00 (e.g., 09:00 = 32400 seconds)
    int light_id;
    int car_count;
};

/**
 *  Structure to hold the partial aggregation result from a Slave.
 * Slaves gather a vector of these back to the Master.
 */
struct AggregatedData {
    int light_id;
    long long total_cars;
};


// --- Global MPI Variables (Initialised in main) ---
MPI_Datatype MPI_TRAFFIC_DATA;
MPI_Datatype MPI_AGGREGATED_DATA;


// --- Helper Functions ---


int parse_time_to_seconds(const std::string& time_str) {
    if (time_str.length() != 5 || time_str[2] != ':') {
        // Fallback or error handling for invalid format
        std::cerr << "Error: Invalid time format encountered: " << time_str << std::endl;
        return 0; 
    }
    
    try {
        // Extract hours and minutes
        int hours = std::stoi(time_str.substr(0, 2));
        int minutes = std::stoi(time_str.substr(3, 2));
        
        // Calculate total seconds
        return (hours * 3600) + (minutes * 60);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing time components: " << e.what() << " for " << time_str << std::endl;
        return 0;
    }
}


void create_mpi_traffic_data_type() {
    int block_lengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
    MPI_Aint offsets[3];

    // Calculate offsets manually as C++ struct padding might be complex
    offsets[0] = offsetof(TrafficData, timestamp);
    offsets[1] = offsetof(TrafficData, light_id);
    offsets[2] = offsetof(TrafficData, car_count);

    MPI_Type_create_struct(3, block_lengths, offsets, types, &MPI_TRAFFIC_DATA);
    MPI_Type_commit(&MPI_TRAFFIC_DATA);
}


void create_mpi_aggregated_data_type() {
    int block_lengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_INT, MPI_LONG_LONG};
    MPI_Aint offsets[2];

    offsets[0] = offsetof(AggregatedData, light_id);
    offsets[1] = offsetof(AggregatedData, total_cars);

    MPI_Type_create_struct(2, block_lengths, offsets, types, &MPI_AGGREGATED_DATA);
    MPI_Type_commit(&MPI_AGGREGATED_DATA);
}


// --- MPI Roles ---

std::map<int, std::vector<TrafficData>> master_read_and_group(const std::string& filename) {
    std::map<int, std::vector<TrafficData>> hourly_groups;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open data file: " << filename << std::endl;
        // In a real scenario, this should use MPI_Abort
        return hourly_groups;
    }

    std::cout << "Master: Reading and grouping data by hour..." << std::endl;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue; // Skip comments and empty lines

        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> parts;
        
        // Split line by comma
        while (std::getline(ss, segment, ',')) {
            parts.push_back(segment);
        }

        if (parts.size() != 3) {
            std::cerr << "Warning: Skipping malformed line: " << line << std::endl;
            continue;
        }

        // --- FIX: Parse HH:MM string to seconds ---
        std::string time_str = parts[0];
        int timestamp_sec = parse_time_to_seconds(time_str);
        
        if (timestamp_sec == 0 && time_str != "00:00") {
            // Error occurred during parsing, skip or handle
            continue;
        }
        // --- END FIX ---

        try {
            int light_id = std::stoi(parts[1]);
            int car_count = std::stoi(parts[2]);

            // Calculate the hour index (grouping records that fall within the same 3600s block)
            // e.g., 09:00 (32400s) / 3600 = 9 (Hour Index 9)
            int hour_index = (timestamp_sec / 3600); 

            // Add data to the corresponding hourly group
            TrafficData data = {timestamp_sec, light_id, car_count};
            hourly_groups[hour_index].push_back(data);

        } catch (const std::exception& e) {
            std::cerr << "Warning: Data conversion error on line '" << line << "': " << e.what() << std::endl;
        }
    }
    file.close();
    return hourly_groups;
}



void analyze_congestion(const std::vector<AggregatedData>& all_aggregated_data, int N, int hour) {
    std::map<int, long long> final_map;

    // 1. Re-aggregate the partial results gathered from all Slaves
    for (const auto& item : all_aggregated_data) {
        final_map[item.light_id] += item.total_cars;
    }

    // 2. Convert map to vector for sorting
    std::vector<AggregatedData> sorted_results;
    for (const auto& pair : final_map) {
        sorted_results.push_back({pair.first, pair.second});
    }

    // 3. Sort by total_cars in descending order (using in-memory sort)
    std::sort(sorted_results.begin(), sorted_results.end(), 
        [](const AggregatedData& a, const AggregatedData& b) {
            return a.total_cars > b.total_cars;
        });

    // 4. Print the final report
    std::cout << "\n========================================================" << std::endl;
    
    // Explicitly set fill character to space to prevent '0' or large invisible characters
    // from padding the output when using setw for the title line.
    std::cout << std::setfill(' '); 
    
    // Print the title line with simple hour number and indentation for clean visual alignment
    std::cout << "   " << "Top " << N << " Congested Traffic Lights - Hour " << hour << std::endl;
              
    std::cout << "========================================================" << std::endl;
    
    // Use left alignment and fixed widths for clean column separation.
    std::cout << std::left << std::setw(15) << "Rank"
              << std::setw(20) << "Traffic Light ID"
              << std::setw(20) << "Total Cars Passed" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    for (int i = 0; i < std::min((int)sorted_results.size(), N); ++i) {
        std::cout << std::left << std::setw(15) << (i + 1)
                  << std::setw(20) << sorted_results[i].light_id
                  << std::setw(20) << sorted_results[i].total_cars << std::endl;
    }
    std::cout << "========================================================" << std::endl;
}

/**
 * @brief Slave (Rank > 0) function to process its local chunk of data.
 * * @param local_data A vector of TrafficData records received by the Slave.
 * @return std::map<int, long long> Local congestion aggregation map.
 */
std::map<int, long long> process_local_data(const std::vector<TrafficData>& local_data) {
    std::map<int, long long> local_congestion_map;
    
    // Simple aggregation: sum car_count for each light_id
    for (const auto& data : local_data) {
        local_congestion_map[data.light_id] += data.car_count;
    }
    
    return local_congestion_map;
}


// --- Main MPI Execution ---

void run_mpi_traffic_simulator(int N, int world_rank, int world_size, const std::string& filename) {
    std::map<int, std::vector<TrafficData>> hourly_groups;
    std::vector<int> hour_indices_to_process; // NEW: Vector to store the actual hour indices (e.g., 9, 10, 11)
    int total_hours = 0;

    // Master (Rank 0) reads and groups the data
    if (world_rank == 0) {
        hourly_groups = master_read_and_group(filename);
        
        // FIX 1: Extract actual keys from the map and sort them for consistent iteration
        for (const auto& pair : hourly_groups) {
            hour_indices_to_process.push_back(pair.first);
        }
        std::sort(hour_indices_to_process.begin(), hour_indices_to_process.end());
        total_hours = hour_indices_to_process.size(); // total_hours is now the number of groups
        // END FIX 1
        
        std::cout << "Master: Successfully grouped data into " << total_hours << " hour(s)." << std::endl;
    }

    // Broadcast the total number of hours (number of keys) to all Slaves
    MPI_Bcast(&total_hours, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // If no data was read or grouped, terminate gracefully
    if (total_hours == 0) {
        if (world_rank == 0) {
            std::cerr << "Error: No valid data found for analysis." << std::endl;
        }
        return;
    }
    
    // FIX 2: Slaves must resize their vector and receive the actual hour indices.
    hour_indices_to_process.resize(total_hours);
    MPI_Bcast(hour_indices_to_process.data(), total_hours, MPI_INT, 0, MPI_COMM_WORLD);
    // END FIX 2

    // The Master and Slaves loop through each hour synchronously.
    // Loop over the count of the groups (total_hours)
    for (int i = 0; i < total_hours; ++i) {
        
        // FIX 3: Get the actual hour index (e.g., 9, 10, 11) to use as the reporting hour
        int current_hour_index = hour_indices_to_process[i];

        std::vector<TrafficData> master_data_buffer;
        int total_records_for_hour = 0;
        int *sendcounts = nullptr;
        int *displs = nullptr;

        if (world_rank == 0) {
            // Master prepares data for the current hour using the actual index key
            // Use the actual hour index (e.g., 9) to retrieve data from the map.
            master_data_buffer = hourly_groups.at(current_hour_index);
            
            total_records_for_hour = master_data_buffer.size();

            // Calculate sendcounts and displacements for the current hour
            sendcounts = new int[world_size];
            displs = new int[world_size];
            int records_per_process = total_records_for_hour / world_size;
            int remainder = total_records_for_hour % world_size;
            int current_displacement = 0;

            for (int j = 0; j < world_size; ++j) {
                sendcounts[j] = records_per_process + (j < remainder ? 1 : 0);
                displs[j] = current_displacement;
                current_displacement += sendcounts[j];
            }
        }

        // 1. SCATTER the size of the receive buffer to each process (robustness fix)
        int local_receive_count = 0;
        MPI_Scatter(sendcounts, 1, MPI_INT, &local_receive_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 2. Slaves allocate memory based on the received count
        std::vector<TrafficData> slave_local_data(local_receive_count);

        // 3. SCATTERV the actual data chunk for the current hour
        MPI_Scatterv(
            master_data_buffer.data(), // Send buffer (only significant at root)
            sendcounts,                 // Number of elements to send to each process (at root)
            displs,                     // Displacement for each block (at root)
            MPI_TRAFFIC_DATA,           // Send type
            slave_local_data.data(),    // Receive buffer
            local_receive_count,        // Receive count (local)
            MPI_TRAFFIC_DATA,           // Receive type
            0,                          // Root
            MPI_COMM_WORLD
        );

        // 4. Local Processing (Computation)
        std::map<int, long long> slave_map = process_local_data(slave_local_data);
        
        // Convert the map to a vector for MPI_Gatherv
        std::vector<AggregatedData> slave_results;
        for (const auto& pair : slave_map) {
            slave_results.push_back({pair.first, pair.second});
        }
        int slave_result_size = slave_results.size();

        // 5. Result Collection (Gatherv Coordination)
        // Master and Slaves participate in getting the size of all result vectors
        int *recvcounts_results = new int[world_size];
        MPI_Allgather(&slave_result_size, 1, MPI_INT, recvcounts_results, 1, MPI_INT, MPI_COMM_WORLD);

        // 6. Calculate displacements for Gatherv
        int total_results_size = 0;
        int *rdispls_results = new int[world_size];
        rdispls_results[0] = 0;
        total_results_size += recvcounts_results[0];
        
        for (int j = 1; j < world_size; ++j) {
            rdispls_results[j] = rdispls_results[j - 1] + recvcounts_results[j - 1];
            total_results_size += recvcounts_results[j];
        }

        // 7. GATHERV the partial results to the Master
        std::vector<AggregatedData> all_aggregated_data;
        if (world_rank == 0) {
            all_aggregated_data.resize(total_results_size);
        }

        MPI_Gatherv(
            slave_results.data(),       // Send buffer
            slave_result_size,          // Send count
            MPI_AGGREGATED_DATA,        // Send type
            all_aggregated_data.data(), // Receive buffer (Master only)
            recvcounts_results,         // Receive count for each process (Master only)
            rdispls_results,            // Displacements (Master only)
            MPI_AGGREGATED_DATA,        // Receive type
            0,                          // Root
            MPI_COMM_WORLD
        );

        // 8. Master analyzes and reports the results for this specific hour
        if (world_rank == 0) {
            analyze_congestion(all_aggregated_data, N, current_hour_index); 
        }

        // Cleanup for the current hour iteration
        delete[] recvcounts_results;
        delete[] rdispls_results;
        if (world_rank == 0) {
            delete[] sendcounts;
            delete[] displs;
        }
    }
}


// --- Main Function ---

int main(int argc, char* argv[]) {
    // 1. Initialize MPI
    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        std::cout << "Starting MPI Traffic Simulator with " << world_size << " processes." << std::endl;
    }

    // 2. Argument Parsing (Check for filename and N)
    if (argc != 3) {
        if (world_rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <data_file.txt> <N_top_lights>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string filename = argv[1];
    int N = std::stoi(argv[2]);

    if (N <= 0) {
        if (world_rank == 0) {
            std::cerr << "Error: N must be a positive integer." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // 3. Create Custom MPI Datatypes
    create_mpi_traffic_data_type();
    create_mpi_aggregated_data_type();

    // 4. Run the simulation
    run_mpi_traffic_simulator(N, world_rank, world_size, filename);

    // 5. Cleanup
    MPI_Type_free(&MPI_TRAFFIC_DATA);
    MPI_Type_free(&MPI_AGGREGATED_DATA);
    MPI_Finalize();

    return 0;
}
