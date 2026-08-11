// Chapter 4 — The Map pattern
// Problem 4.3: Vector ReLU
// https://swforces.com/problem/195
//
// The kernel is printed in the book. The main() is not printed there; it
// follows the same five-step pattern as Vector Addition, using float instead
// of int. The judge compares answers with a small tolerance, so the exact
// print format of the real numbers doesn't matter.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void reluKernel(float* A, float* B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        B[idx] = (A[idx] > 0.0f) ? A[idx] : 0.0f;
    }
}

int main() {
    // Read input
    int n;
    scanf("%d", &n);
    float *h_A = (float*)malloc(n * sizeof(float));
    float *h_B = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) scanf("%f", &h_A[i]);

    // Step 1: Allocate GPU memory
    float *d_A, *d_B;
    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));

    // Step 2: Copy data from CPU to GPU
    cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);

    // Step 3: Launch kernel
    reluKernel<<<(n + 255) / 256, 256>>>(d_A, d_B, n);

    // Step 4: Copy results back and print
    cudaMemcpy(h_B, d_B, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) printf("%g ", h_B[i]);

    // Step 5: Free memory
    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B);
    return 0;
}
