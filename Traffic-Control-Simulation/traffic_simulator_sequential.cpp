#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <chrono> // For timing the execution

// --- 1. Data Structures ---

// Defines the data structure for a single traffic signal measurement.
struct TrafficData {
    std::string timestamp; // e.g., "10:05"
    int light_id;          // Unique ID for the traffic light
    int car_count;         // Number of cars recorded in the 5-minute interval
};

// --- 2. Data Processing Function ---

// Purpose: Reads all data from the input file line-by-line, parses it, and
// aggregates car counts into a single congestion map. This performs the work
// of both the Producer and Consumer functions sequentially.
std::map<int, long long> process_sequential_data(const std::string& filename) {
    std::map<int, long long> congestion_map;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open data file: " << filename << std::endl;
        return congestion_map;
    }

    // Read the file line by line
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string ts_str, id_str, count_str;
        
        // Data format: timestamp,light_id,car_count
        if (std::getline(ss, ts_str, ',') &&
            std::getline(ss, id_str, ',') &&
            std::getline(ss, count_str, ',')) {

            try {
                int light_id = std::stoi(id_str);
                int car_count = std::stoi(count_str);
                
                // Direct aggregation (no lock needed in sequential context)
                congestion_map[light_id] += car_count;
                
            } catch (const std::exception& e) {
                std::cerr << "Sequential Error: Invalid data format in line: " << line << std::endl;
                continue;
            }
        }
    }
    return congestion_map;
}

// --- 3. Analysis and Report Generation ---

// Purpose: Finds and prints the top N most congested traffic lights from the provided map.
void analyze_congestion(const std::map<int, long long>& congestion_map, int top_n) {
    // 1. Convert map to vector of pairs for sorting
    std::vector<std::pair<long long, int>> sorted_lights; // pair: (count, light_id)
    
    for (const auto& pair : congestion_map) {
        sorted_lights.push_back({pair.second, pair.first});
    }

    if (sorted_lights.empty()) {
        std::cout << "\nNo traffic data was processed." << std::endl;
        return;
    }

    // 2. Sort the vector (descending by car count)
    std::sort(sorted_lights.rbegin(), sorted_lights.rend());

    // 3. Print the top N
    std::cout << "\n========================================================" << std::endl;
    std::cout << "          Top " << top_n << " Most Congested Traffic Lights" << std::endl;
    std::cout << "========================================================" << std::endl;
    std::cout << std::left << std::setw(15) << "Rank" 
              << std::setw(20) << "Traffic Light ID" 
              << std::setw(20) << "Total Cars Passed" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    int rank = 1;
    for (const auto& pair : sorted_lights) {
        if (rank > top_n) break;
        std::cout << std::left << std::setw(15) << rank++
                  << std::setw(20) << pair.second 
                  << std::setw(20) << pair.first << std::endl;
    }
    std::cout << "========================================================" << std::endl;
}

// --- 4. Main Execution ---

int main(int argc, char* argv[]) {
    // Usage requires file and top_N only (no threads needed)
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <data_file> <top_N>" << std::endl;
        std::cerr << "Example: " << argv[0] << " traffic_data.txt 5" << std::endl;
        return 1;
    }

    const std::string data_file = argv[1];
    const int top_n = std::atoi(argv[2]);

    std::cout << "Starting Sequential Traffic Simulation (Microsecond Timing)..." << std::endl;

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Process all data sequentially
    std::map<int, long long> final_congestion_map = process_sequential_data(data_file);

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    // CHANGED: Using microseconds for higher precision
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    // Final Analysis
    analyze_congestion(final_congestion_map, top_n);

    std::cout << "\nSimulation Complete." << std::endl;
    // CHANGED: Printing time in microseconds
    std::cout << "Sequential Execution Time: " << duration.count() << " microseconds." << std::endl;

    return 0;
}
