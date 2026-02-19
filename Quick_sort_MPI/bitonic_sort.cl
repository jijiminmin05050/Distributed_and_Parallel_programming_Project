// Purpose: Implements the compare and exchange step of the Bitonic Sort algorithm.
// Execution: Runs on the OpenCL device (GPU). Each work item (thread) compares and
//            potentially swaps two elements in the array.

__kernel void bitonic_sort(__global int* array,
                           const int m, // Current size of the merging sequence (power of 2)
                           const int j, // Current distance between compared elements (power of 2)
                           const int n_padded) {
    
    // Global index: determines the index of the first element in the comparison pair.
    // get_global_id(0) runs from 0 to (n_padded / 2) - 1
    const int i = get_global_id(0);
    
    // k is the index of the first element in the pair
    const int k = i + (i / j) * j;

    // Index of the two elements to compare/swap
    const int idx1 = k;
    const int idx2 = k + j;
    
    // Only process valid indices within the padded array bounds
    if (idx2 < n_padded) {
        // Determine the direction of the sort (ascending or descending)
        // If (k & m) is non-zero, the sequence is sorted in descending order (m=2, 4, 8, ...)
        const int dir = (k & m); // 0 or non-zero value
        
        const int val1 = array[idx1];
        const int val2 = array[idx2];
        
        bool swap = (val1 > val2);

        // If 'dir' is non-zero, we swap if val1 is LESS THAN val2 (descending)
        if (dir != 0) {
            swap = (val1 < val2);
        }

        if (swap) {
            array[idx1] = val2;
            array[idx2] = val1;
        }
    }
}
