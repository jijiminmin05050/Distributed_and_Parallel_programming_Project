#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <chrono> // Added for timing

// --- 1. Data Structures ---

// Defines the data structure for a single traffic signal measurement.
struct TrafficData {
    std::string timestamp; // e.g., "10:05"
    int light_id;          // Unique ID for the traffic light
    int car_count;         // Number of cars recorded in the 5-minute interval
};

// Global parameters for the bounded buffer
constexpr int BUFFER_CAPACITY = 50;
bool g_done_reading = false; // Flag to signal consumers that producers are done

// Shared resources, requiring synchronization
std::queue<TrafficData> g_buffer;
std::mutex g_buffer_mutex;
std::condition_variable g_cond_producer; // Condition for buffer NOT full
std::condition_variable g_cond_consumer; // Condition for buffer NOT empty

// Shared resource for congestion tracking
// Key: traffic_light_id, Value: total_cars_passed
std::map<int, long long> g_congestion_map;
std::mutex g_congestion_mutex;

// --- 2. Traffic Producer Function ---

// Purpose: Reads data from the input file line-by-line and pushes parsed TrafficData
//          objects into the shared bounded buffer.
void traffic_producer(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Producer Error: Could not open data file: " << filename << std::endl;
        return;
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

            TrafficData data;
            data.timestamp = ts_str;
            try {
                data.light_id = std::stoi(id_str);
                data.car_count = std::stoi(count_str);
            } catch (const std::exception& e) {
                std::cerr << "Producer Error: Invalid data format in line: " << line << std::endl;
                continue;
            }

            // --- Bounded Buffer Logic (Blocking Push) ---
            {
                std::unique_lock<std::mutex> lock(g_buffer_mutex);
                
                // Block if the buffer is full
                g_cond_producer.wait(lock, []{ return g_buffer.size() < BUFFER_CAPACITY; });

                g_buffer.push(data);
            } // Lock released

            // Notify one waiting consumer that the buffer is not empty
            g_cond_consumer.notify_one();
        }
    }

    // Note: The final g_done_reading signal is handled in main after all producers join.
}

// --- 3. Traffic Consumer Function ---

// Purpose: Reads data from the bounded buffer and updates the global congestion map
//          by aggregating car counts for each traffic light ID.
void traffic_consumer(int consumer_id) {
    int processed_count = 0;

    while (true) {
        TrafficData data;
        
        // --- Bounded Buffer Logic (Blocking Pop) ---
        {
            std::unique_lock<std::mutex> lock(g_buffer_mutex);

            // Wait if the buffer is empty AND producers are NOT finished
            g_cond_consumer.wait(lock, []{ return !g_buffer.empty() || g_done_reading; });

            // Exit condition: Buffer is empty AND all producers are done reading
            if (g_buffer.empty() && g_done_reading) {
                // Important: Notify other consumers to unblock and exit gracefully
                g_cond_consumer.notify_all(); 
                break;
            }

            data = g_buffer.front();
            g_buffer.pop();
        } // Lock released

        // Notify one waiting producer that the buffer is not full
        g_cond_producer.notify_one();

        // --- Congestion Map Update (Non-Blocking) ---
        {
            std::lock_guard<std::mutex> map_lock(g_congestion_mutex);
            // Update the total car count for the specific traffic light ID
            g_congestion_map[data.light_id] += data.car_count;
        } // Lock released

        processed_count++;
    }
}

// --- 4. Analysis and Report Generation ---

// Purpose: Finds and prints the top N most congested traffic lights from the global map.
void analyze_congestion(int top_n) {
    // 1. Convert map to vector of pairs for sorting
    std::vector<std::pair<long long, int>> sorted_lights; // pair: (count, light_id)
    
    // Acquire the lock to safely read the final map contents
    std::lock_guard<std::mutex> map_lock(g_congestion_mutex); 
    
    for (const auto& pair : g_congestion_map) {
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

// --- 5. Main Execution ---

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <data_file> <num_producers> <num_consumers> <top_N>" << std::endl;
        std::cerr << "Example: " << argv[0] << " traffic_data.txt 1 4 5" << std::endl;
        return 1;
    }

    const std::string data_file = argv[1];
    const int num_producers = std::atoi(argv[2]);
    const int num_consumers = std::atoi(argv[3]);
    const int top_n = std::atoi(argv[4]);

    if (num_producers <= 0 || num_consumers <= 0 || top_n <= 0) {
        std::cerr << "Error: Number of threads and Top N must be positive." << std::endl;
        return 1;
    }

    std::cout << "Starting Multi-Threaded Traffic Simulation with:\n"
              << "  Producers: " << num_producers << "\n"
              << "  Consumers: " << num_consumers << "\n"
              << "  Buffer Capacity: " << BUFFER_CAPACITY << std::endl;

    // --- Start Timing ---
    auto start_time = std::chrono::high_resolution_clock::now();

    // Start Producer Threads
    std::vector<std::thread> producer_threads;
    for (int i = 0; i < num_producers; ++i) {
        producer_threads.emplace_back(traffic_producer, data_file);
    }

    // Start Consumer Threads
    std::vector<std::thread> consumer_threads;
    for (int i = 0; i < num_consumers; ++i) {
        consumer_threads.emplace_back(traffic_consumer, i + 1);
    }

    // Wait for all producer threads to finish reading the data file
    for (auto& t : producer_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Signal to consumers that all production is complete
    {
        std::lock_guard<std::mutex> lock(g_buffer_mutex);
        g_done_reading = true;
    }
    // Wake up all waiting consumers so they can check the exit condition
    g_cond_consumer.notify_all(); 

    // Wait for all consumer threads to finish processing
    for (auto& t : consumer_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    // --- Stop Timing ---
    auto end_time = std::chrono::high_resolution_clock::now();
    // Changed resolution to microseconds for finer detail
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); 

    // Final Analysis
    analyze_congestion(top_n);

    std::cout << "\nSimulation Complete." << std::endl;
    std::cout << "Multi-Threaded Execution Time: " << duration.count() << " microseconds." << std::endl;

    return 0;
}
