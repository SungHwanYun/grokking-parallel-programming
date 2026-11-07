// Chapter 13 — Read-only data optimization
// Problem 13.1: Scale array with constant factor
// https://swforces.com/problem/196
//
// The submittable version of 01-scale-array-constant.cu, changed exactly as
// the chapter describes: the scale factor, N, and the array are read from
// standard input instead of being hard-coded, and the results are printed
// with %g rather than %.1f. The kernel and the three constant-memory steps
// are the same.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Step 1: Declare in constant memory (file scope)
__constant__ float scaleFactor;

__global__ void scaleKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Step 3: Use it (every thread reads the same value)
        data[idx] = data[idx] * scaleFactor;    // Broadcast
    }
}

int main() {
    float h_scale;
    int n;
    scanf("%f", &h_scale);
    scanf("%d", &n);
    float* h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) scanf("%f", &h_data[i]);

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
        printf("%g ", h_data[i]);
    printf("\n");

    cudaFree(d_data);
    free(h_data);
    return 0;
}
