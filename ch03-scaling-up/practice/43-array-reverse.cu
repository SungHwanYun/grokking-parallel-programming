// Chapter 3 — Practice on swforces
// Problem 43: Array Reverse
// https://swforces.com/problem/43
//
// Reference solution (not printed in the book). Thread idx writes
// B[idx] = A[n - 1 - idx]. Reading A and writing B (two arrays) keeps every
// thread independent; reversing one array in place would make thread idx
// overwrite the element that thread n - 1 - idx still has to read.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void reverseKernel(int* A, int* B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        B[idx] = A[n - 1 - idx];
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_data[i]);

    int *d_A, *d_B;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMemcpy(d_A, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    reverseKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_B, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) printf("%d ", h_data[i]);
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_data);
    return 0;
}
