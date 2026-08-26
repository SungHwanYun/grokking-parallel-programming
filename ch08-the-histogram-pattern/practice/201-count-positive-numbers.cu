// Chapter 8 — Practice on swforces
// Problem 201: Count Positive Numbers
// https://swforces.com/problem/201
//
// Reference solution (not printed in the book). The kernel is the chapter's
// countPositive unchanged; only main() differs, reading the array from
// standard input. n can be up to 2,048, so the launch really does use
// multiple blocks, all incrementing the same counter through atomicAdd.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void countPositive(int* data, int* count, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && data[idx] > 0) {
        atomicAdd(count, 1);
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_data[i]);
    int hostCount = 0;

    int *d_data, *d_count;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &hostCount, sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    countPositive<<<blocks, threadsPerBlock>>>(d_data, d_count, n);

    cudaMemcpy(&hostCount, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", hostCount);

    cudaFree(d_data);
    cudaFree(d_count);
    free(h_data);
    return 0;
}
