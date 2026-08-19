// Chapter 2 — Practice on swforces
// Problem 197: Double Array Elements
// https://swforces.com/problem/197
//
// Reference solution (not printed in the book). Per the problem statement,
// launch one thread per element in a single block, and have each thread
// double its own element. Same five-step pattern as the squaring program.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void doubleKernel(int* data, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        scanf("%d", &h_data[i]);

    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    doubleKernel<<<1, n>>>(d_data, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        printf("%d ", h_data[i]);

    cudaFree(d_data);
    free(h_data);
    return 0;
}
