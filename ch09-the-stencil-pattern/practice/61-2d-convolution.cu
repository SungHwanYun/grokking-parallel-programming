// Chapter 9 — Practice on swforces
// Problem 61: 2D Convolution
// https://swforces.com/problem/61
//
// Reference solution (not printed in the book). A weighted stencil in
// valid mode: output (i, j) is the weighted sum of the m x m window of A
// whose top-left corner is (i, j), with the kernel matrix B as the
// weights. Because the output is only (n - m + 1) square, every window
// lies inside A and no boundary strategy is needed. One thread per output
// cell, launched as a 2D grid of 16 x 16 blocks like the chapter's blur2D.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void convolve2D(int* A, int* B, int* C, int n, int m) {
    int outSize = n - m + 1;
    int x = blockIdx.x * blockDim.x + threadIdx.x;    // column of C
    int y = blockIdx.y * blockDim.y + threadIdx.y;    // row of C
    if (x >= outSize || y >= outSize) return;

    int sum = 0;
    for (int p = 0; p < m; p++) {
        for (int q = 0; q < m; q++) {
            sum += A[(y + p) * n + (x + q)] * B[p * m + q];    // Multiply by the weight before adding
        }
    }
    C[y * outSize + x] = sum;
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    int outSize = n - m + 1;
    int* h_A = (int*)malloc(n * n * sizeof(int));
    int* h_B = (int*)malloc(m * m * sizeof(int));
    int* h_C = (int*)malloc(outSize * outSize * sizeof(int));
    for (int i = 0; i < n * n; i++) scanf("%d", &h_A[i]);
    for (int i = 0; i < m * m; i++) scanf("%d", &h_B[i]);

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(int));
    cudaMalloc(&d_B, m * m * sizeof(int));
    cudaMalloc(&d_C, outSize * outSize * sizeof(int));
    cudaMemcpy(d_A, h_A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, m * m * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((outSize + 15) / 16, (outSize + 15) / 16);
    convolve2D<<<grid, block>>>(d_A, d_B, d_C, n, m);

    cudaMemcpy(h_C, d_C, outSize * outSize * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < outSize; i++) {
        for (int j = 0; j < outSize; j++) printf("%d ", h_C[i * outSize + j]);
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
