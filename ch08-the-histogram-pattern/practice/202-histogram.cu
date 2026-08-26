// Chapter 8 — Practice on swforces
// Problem 202: Histogram
// https://swforces.com/problem/202
//
// Reference solution (not printed in the book). The kernel is the chapter's
// histogramShared unchanged: each block builds a private histogram in shared
// memory, then merges it into the global one with one atomicAdd per non-empty
// bin. main() reads the array, zeroes the 256 global bins, and prints them.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void histogramShared(int* data, int* hist, int n) {
    __shared__ int localHist[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Initialize
    for (int i = tid; i < 256; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Step 2: Build local histogram
    if (idx < n) {
        int bin = data[idx];
        atomicAdd(&localHist[bin], 1);
    }
    __syncthreads();

    // Step 3: Merge to global
    for (int i = tid; i < 256; i += blockDim.x) {
        if (localHist[i] > 0) atomicAdd(&hist[i], localHist[i]);
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_data[i]);

    int *d_data, *d_hist;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_hist, 256 * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 256 * sizeof(int));  // Zero the bins before launch

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    histogramShared<<<blocks, threadsPerBlock>>>(d_data, d_hist, n);

    int h_hist[256];
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 256; i++) printf("%d ", h_hist[i]);
    printf("\n");

    cudaFree(d_data);
    cudaFree(d_hist);
    free(h_data);
    return 0;
}
