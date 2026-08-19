// Chapter 3 — Scaling up: from one block to many
// Problem 3.1: Array Double
// https://swforces.com/problem/192
//
// The chapter's four-step thread-to-element recipe, annotated inline as Steps 1-4.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void doubleKernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Step 3: global index
    if (idx < n) {                                    // Step 4: bounds check
        data[idx] = data[idx] * 2;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int size = n * sizeof(int);

    // Allocate host (CPU) memory and initialize
    int* h_data = (int*)malloc(size);                 // h_ prefix = host (CPU)
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_data[i]);
    }

    // Allocate device (GPU) memory and copy data to GPU
    int* d_data;                                      // d_ prefix = device (GPU)
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;                        // Step 1: block size
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;  // Step 2: ceiling division
    doubleKernel<<<numBlocks, threadsPerBlock>>>(d_data, n);      // Launch!
    cudaDeviceSynchronize();                          // Wait for GPU to finish

    // Copy results back to CPU
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        printf("%d ", h_data[i]);
    }

    // Cleanup
    cudaFree(d_data);
    free(h_data);
    return 0;
}
