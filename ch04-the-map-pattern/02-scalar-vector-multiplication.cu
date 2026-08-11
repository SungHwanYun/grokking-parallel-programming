// Chapter 4 — The Map pattern
// Problem 4.2: Scalar Vector Multiplication
// https://swforces.com/problem/39
//
// The kernel is printed in the book. The main() is not printed there; as the
// book notes, it follows the same five-step pattern as Vector Addition, with
// these differences: read one scalar k and one array A (instead of two
// arrays), allocate GPU memory for A and the output B, and pass k directly
// as a kernel argument (scalars need no cudaMalloc/cudaMemcpy).

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void scalarMultiplyKernel(int* A, int* B, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        B[idx] = k * A[idx];
    }
}

int main() {
    // Read input: n, k, then vector A
    int n, k;
    scanf("%d %d", &n, &k);
    int *h_A = (int*)malloc(n * sizeof(int));
    int *h_B = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_A[i]);

    // Step 1: Allocate GPU memory (input A and output B)
    int *d_A, *d_B;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));

    // Step 2: Copy data from CPU to GPU (k travels as a kernel argument)
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);

    // Step 3: Launch kernel
    scalarMultiplyKernel<<<(n + 255) / 256, 256>>>(d_A, d_B, k, n);

    // Step 4: Copy results back and print
    cudaMemcpy(h_B, d_B, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) printf("%d ", h_B[i]);

    // Step 5: Free memory
    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B);
    return 0;
}
