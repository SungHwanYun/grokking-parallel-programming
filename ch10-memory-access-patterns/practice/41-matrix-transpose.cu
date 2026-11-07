// Chapter 10 — Practice on swforces
// Problem 41: Matrix Transpose
// https://swforces.com/problem/41
//
// Reference solution (not printed in the book). One thread per element
// of the n x m input, in a 2D grid: thread (x, y) reads A[y][x] and writes
// B[x][y]. The access pattern is the whole point. Threads with consecutive
// x read consecutive addresses of A (coalesced) but write addresses n
// apart in B (strided), so one side of a plain transpose is always
// strided. The matrices here are tiny; chapter 11's tiled transpose shows
// how shared memory makes both sides coalesced on large ones.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void transposeKernel(int* A, int* B, int n, int m) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;    // column of A
    int y = blockIdx.y * blockDim.y + threadIdx.y;    // row of A
    if (x < m && y < n) {
        B[x * n + y] = A[y * m + x];    // Coalesced read, strided write
    }
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    int* h_A = (int*)malloc(n * m * sizeof(int));
    int* h_B = (int*)malloc(n * m * sizeof(int));
    for (int i = 0; i < n * m; i++) scanf("%d", &h_A[i]);

    int *d_A, *d_B;
    cudaMalloc(&d_A, n * m * sizeof(int));
    cudaMalloc(&d_B, n * m * sizeof(int));
    cudaMemcpy(d_A, h_A, n * m * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((m + 15) / 16, (n + 15) / 16);
    transposeKernel<<<grid, block>>>(d_A, d_B, n, m);

    cudaMemcpy(h_B, d_B, n * m * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) printf("%d ", h_B[i * n + j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
    return 0;
}
