Project Report: Parallel Traffic Congestion Analysis (MPI + OpenMP Hybrid)
Grade Level Assessment: Undergraduate/Graduate Computer Science (High-Performance Computing)

1. Project Scope
1.1 Problem Statement
Urban traffic management requires rapid analysis of congestion points to optimize signal timing and resource allocation. Traditional sequential data processing methods become bottlenecks when dealing with high-volume, real-time traffic sensor data. The goal is to develop a hybrid parallel computing solution that efficiently processes large datasets of traffic light activity to identify the top N most congested intersections.

1.2 Objectives
Implement a parallel solution using MPI to distribute the workload across multiple processes (inter-node or inter-core parallelism).

Utilize OpenMP to introduce multithreading within each MPI process, optimizing local data aggregation (intra-node parallelism).

Design and implement a robust data aggregation strategy using MPI collective communications (MPI_Scatterv, MPI_Gatherv).

Generate a clear, sorted report of the top N most congested traffic lights.

1.3 Key Performance Indicators (KPIs)
Scalability: Achieve significant speedup when running with 2 or more MPI processes compared to a single-process run.

Correctness: The final aggregated car counts must be accurate, irrespective of the number of processes/threads used.

Robustness: The solution must gracefully handle uneven data distribution and varying file sizes.

2. Design
2.1 Architecture: MPI + OpenMP Hybrid Model
The project uses a standard Hybrid Parallel Architecture suitable for modern cluster computing:

MPI (Inter-Process): Used for coarse-grained communication and load distribution across different memory spaces. A single Master (Rank 0) coordinates the entire process, while Slaves (Rank > 0) perform parallel processing.

OpenMP (Intra-Process): Used for fine-grained parallelism within the memory space of each individual MPI rank.

2.2 Data Flow and Communication Strategy
Reading and Distribution (Master): Rank 0 reads the entire data file and calculates the precise sendcounts and displs arrays for dynamic, balanced distribution. The total record count is broadcast using MPI_Bcast.

Local Parallel Processing (All Ranks): Each rank receives its data chunk. Inside the process_local_data function, OpenMP is employed to use thread-local maps for concurrent aggregation, which are then merged into a single local map by the master thread.

Collection and Aggregation:

Each rank converts its local aggregation map (Light ID → Total Cars) into a flat AggregatedData vector.

MPI_Allgather is used to exchange the size of these vectors (number of unique Light IDs found locally) among all ranks, calculating the global recvcounts and rdispls.

MPI_Gatherv collects all fragmented results back to the Master (Rank 0).

Final Analysis (Master): Rank 0 performs the final consolidation of all collected AggregatedData records (summing counts for duplicate IDs) and sorts them to find the top N.

2.3 Custom MPI Datatype
A custom MPI structure, MPI_AGGREGATED_DATA, is created using MPI_Type_create_struct and offsetof to ensure the C++ struct fields (int and long long) are correctly mapped to their corresponding MPI types (MPI_INT and MPI_LONG_LONG) for reliable transmission.

3. Implementation
The implementation utilized the standard C++ libraries, OpenMPI (or equivalent MPI implementation), and GCC/G++ with the OpenMP flag.

3.1 C++ / OpenMP Implementation Detail
The process_local_data function is the core performance bottleneck and was targeted for optimization:

Instead of a single, shared std::map protected by omp critical sections (which would cause high contention), the implementation uses an array of thread-local maps (std::vector<std::map<int, long long>> thread_maps).

The work loop is parallelized using #pragma omp parallel for.

After the parallel loop, the thread-local maps are merged sequentially, resulting in an efficient, contention-free local aggregation.

3.2 Key MPI Implementation Details
MPI_Bcast for Coordination: The total number of records is broadcast from Rank 0 to all other ranks. This allows every rank to accurately calculate its own expected receive size for the MPI_Scatterv operation, avoiding hardcoding the data size.

MPI_Allgather for Dynamic Gatherv: Since the number of unique traffic lights (the size of the data being returned) is unknown and varies per rank, MPI_Allgather is used to determine the exact size of the final global buffer and the offsets for the MPI_Gatherv operation.

4. Evaluation
4.1 Correctness
Testing with a known, small data set (traffic_data.txt) confirmed the correctness of the final results, regardless of the process count (P=1,2,4). The final car counts for Light ID 101, 205, etc., consistently matched the expected sequential result, confirming the integrity of the data distribution, local aggregation (OpenMP), and global aggregation (MPI) steps.


4.2 Performance (Qualitative)
The hybrid approach offers significant qualitative performance benefits:

MPI: Efficiently distributes the I/O bottleneck (file reading) and the primary processing task (summation) across multiple cores/machines.

OpenMP: Ensures that the time spent processing the local chunk of data (the process_local_data function) is minimized by fully utilizing all available cores on the execution node.

4.3 Future Work
Dynamic Load Balancing: Implement a dynamic work queue instead of static block decomposition to handle cases where light IDs are highly clustered in the data, leading to variable processing times per record.

I/O Parallelism: Implement a parallel file system (e.g., using MPI-IO) to allow all ranks to read their own data chunks simultaneously, eliminating the I/O bottleneck currently present on the Master Rank.

Fault Tolerance: Add logic to handle the failure of a slave process using techniques like checkpointing or recovery protocols.

4.4 Lessons Learned
Hybrid Design Complexity: While highly performant, hybrid MPI/OpenMP requires careful coordination, especially in defining which tasks are parallelized (MPI for data movement, OpenMP for computation).

Non-Contention Mapping: Using thread-local data structures (maps) in OpenMP, followed by a final sequential merge, is dramatically more performant for sparse data aggregation than trying to protect a single, shared data structure.