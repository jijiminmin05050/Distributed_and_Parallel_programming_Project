#include <iostream>
#include <fstream>
#include <cstdlib>
#include <time.h>
#include <chrono>
#include <string>

using namespace std;
using namespace std::chrono;

// Function to initialize matrix with random values
// Assumes a square matrix of size NxN.
void initializeMatrix(int** matrix, int size) {
    // Seed the random generator (only once in main)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = rand() % 100;  // Random values 0-99
        }
    }
}

// Function to allocate memory for NxN matrix
int** allocateMatrix(int size) {
    int** matrix = new int*[size];
    for (int i = 0; i < size; i++) {
        matrix[i] = new int[size];
    }
    return matrix;
}

// Function to deallocate matrix memory
void deallocateMatrix(int** matrix, int size) {
    for (int i = 0; i < size; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// Sequential matrix multiplication: C = A * B
void matrixMultiplication(int** A, int** B, int** C, int size) {
    // Standard IJK order: most cache-friendly when B is stored column-wise,
    // but less so for the current row-major allocation of B.
    // For sequential performance, a loop order optimization (like I K J) 
    // might be beneficial, but we use the standard for simplicity in comparison.
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to write matrix to file
void writeMatrixToFile(int** matrix, int size, const string& filename) {
    // Note: Writing large matrices can take significant time and storage.
    // This function should typically be skipped for large performance runs.
    ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << size << endl; // Write size first
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                outFile << matrix[i][j];
                if (j < size - 1) outFile << " ";
            }
            outFile << endl;
        }
        outFile.close();
        cout << "Result written to " << filename << endl;
    } else {
        cerr << "Error: Could not open file " << filename << endl;
    }
}

int main(int argc, char* argv[]) {
    // Default matrix size is relatively small for quick testing.
    int size = 512;  
    
    // Process command line argument for size
    if (argc > 1) {
        try {
            size = stoi(argv[1]);
        } catch (const invalid_argument& e) {
            cerr << "Invalid matrix size argument: " << argv[1] << endl;
            size = 512;
        } catch (const out_of_range& e) {
            cerr << "Matrix size out of range: " << argv[1] << endl;
            size = 512;
        }
        
        if (size <= 0) {
            cerr << "Matrix size must be positive. Using default size 512." << endl;
            size = 512;
        }
    }
    
    cout << "--- Sequential Matrix Multiplication (Baseline) ---" << endl;
    cout << "Matrix size: " << size << "x" << size << endl;
    
    srand(time(0));
    
    // Allocate and initialize matrices
    int** A = allocateMatrix(size);
    int** B = allocateMatrix(size);
    int** C = allocateMatrix(size);
    
    cout << "Initializing matrices..." << endl;
    initializeMatrix(A, size);
    initializeMatrix(B, size);
    
    // Perform matrix multiplication and measure time
    cout << "Starting matrix multiplication (IJK order)..." << endl;
    auto start = high_resolution_clock::now();
    
    matrixMultiplication(A, B, C, size);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    
    cout << "Matrix multiplication completed!" << endl;
    cout << "Execution time: " << duration.count() << " milliseconds" << endl;
    
    // Optional: Write result to file (uncomment if needed)
    // string filename = "sequential_result_" + to_string(size) + "x" + to_string(size) + ".txt";
    // writeMatrixToFile(C, size, filename);
    
    // Deallocate memory
    deallocateMatrix(A, size);
    deallocateMatrix(B, size);
    deallocateMatrix(C, size);
    
    cout << "--- Baseline Finished ---" << endl;
    
    return 0;
}