// Chapter 4 — The Map pattern
// Problem 4.1: Vector Addition
// https://swforces.com/problem/4
//
// Map with multiple inputs: each thread reads A[idx] and B[idx] and writes
// its own C[idx]. No cudaDeviceSynchronize() is needed before the copy-back:
// a blocking cudaMemcpy waits for all preceding GPU work to finish.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vectorAddKernel(int* A, int* B, int* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Read input
    int n;
    scanf("%d", &n);
    int *h_A = (int*)malloc(n * sizeof(int));
    int *h_B = (int*)malloc(n * sizeof(int));
    int *h_C = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_A[i]);
    for (int i = 0; i < n; i++) scanf("%d", &h_B[i]);

    // Step 1: Allocate GPU memory
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_C, n * sizeof(int));

    // Step 2: Copy data from CPU to GPU
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);

    // Step 3: Launch kernel (enough blocks of 256 threads to cover n)
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // Step 4: Copy results back and print
    cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) printf("%d ", h_C[i]);

    // Step 5: Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
