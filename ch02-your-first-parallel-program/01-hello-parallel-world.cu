// Chapter 2 — Your first parallel program
// Problem 2.1: Print Hello, Parallel World! 8 times
// https://swforces.com/problem/191

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel() {              // Kernel declaration
    printf("Hello, Parallel World!\n");      // The work each thread performs
}

int main() {
    helloKernel<<<1, 8>>>();                 // Launch 8 threads in 1 block
    cudaDeviceSynchronize();                 // Synchronize: wait for GPU
    return 0;
}
