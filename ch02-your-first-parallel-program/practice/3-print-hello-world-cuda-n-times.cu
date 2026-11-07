// Chapter 2 — Practice on swforces
// Problem 3: Print Hello World Cuda N Times
// https://swforces.com/problem/3
//
// Reference solution (not printed in the book). Problem 2.1 with the thread
// count read from input: launch one block of n threads and let every thread
// print the same line. The lines are identical, so the order in which the
// threads reach printf does not matter.

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloKernel() {
    printf("Hello World Cuda\n");
}

int main() {
    int n;
    scanf("%d", &n);
    helloKernel<<<1, n>>>();    // n <= 100, so one block is enough
    cudaDeviceSynchronize();    // Wait for the GPU before the program exits
    return 0;
}
