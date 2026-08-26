// Chapter 13 — Read-Only Data Optimization
// Problem 13.1: Scale Array with Constant Factor
// https://swforces.com/problem/196

#include <stdio.h>    // not printed in the book's listing; needed for printf

// Step 1: Declare in constant memory (file scope)
__constant__ float scaleFactor;

__global__ void scaleKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Step 3: Use it — every thread reads the same value!
        data[idx] = data[idx] * scaleFactor;    // Broadcast
    }
}

int main() {
    int n = 4;
    float h_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_scale = 2.5f;

    // Set up device input
    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float),
               cudaMemcpyHostToDevice);

    // Step 2: Copy scale factor to constant memory
    cudaMemcpyToSymbol(scaleFactor, &h_scale, sizeof(float));


    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    scaleKernel<<<gridSize, blockSize>>>(d_data, n);

    // Copy result back
    cudaMemcpy(h_data, d_data, n * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < n; i++)
        printf("%.1f ", h_data[i]);
    // Output: 2.5 5.0 7.5 10.0

    cudaFree(d_data);
    return 0;
}
