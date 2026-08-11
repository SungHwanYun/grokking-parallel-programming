// Chapter 2 — Your first parallel program
// Problem 2.2: Thread identity
// https://swforces.com/problem/189

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printIndexKernel() {
    int idx = threadIdx.x;                   // Each thread gets its own unique number
    if (idx == 2) {                          // Only thread 2 gets to speak
        printf("Thread %d: Hello!\n", idx);  // Print that number alongside the message
    }
}

int main() {
    printIndexKernel<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
