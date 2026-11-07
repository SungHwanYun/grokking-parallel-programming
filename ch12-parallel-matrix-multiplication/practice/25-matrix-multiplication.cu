// Chapter 12 — Practice on swforces
// Problem 25: Matrix Multiplication
// https://swforces.com/problem/25
//
// Reference solution (not printed in the book). The chapter's matmulNaive
// kernel on integers: one thread per output element, each computing one
// dot product straight from global memory. With M = N = K = n <= 10 the
// whole product fits in a single 16 x 16 block, launched the way the
// chapter launches the naive kernel.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void matmulNaive(int* A, int* B, int* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int* h_A = (int*)malloc(n * n * sizeof(int));
    int* h_B = (int*)malloc(n * n * sizeof(int));
    int* h_C = (int*)malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; i++) scanf("%d", &h_A[i]);
    for (int i = 0; i < n * n; i++) scanf("%d", &h_B[i]);

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(int));
    cudaMalloc(&d_B, n * n * sizeof(int));
    cudaMalloc(&d_C, n * n * sizeof(int));
    cudaMemcpy(d_A, h_A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);                            // 256 threads per block
    dim3 gridDim((n + 15) / 16, (n + 15) / 16);       // ceiling division
    matmulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, n, n, n);

    cudaMemcpy(h_C, d_C, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) printf("%d ", h_C[i * n + j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
