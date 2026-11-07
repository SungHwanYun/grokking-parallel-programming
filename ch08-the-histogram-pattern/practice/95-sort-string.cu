// Chapter 8 — Practice on swforces
// Problem 95: Sort String
// https://swforces.com/problem/95
//
// Reference solution (not printed in the book). Sorting a string of
// lowercase letters is a counting sort, and counting is a histogram: the
// chapter's histogramShared kernel, with each character code as its bin
// (so only bins 'a' through 'z' are ever non-zero), counts the letters,
// and the host writes out each letter as many times as its bin says, in
// alphabetical order. n goes up to 1,000, so several blocks each build a
// private histogram before merging.

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
    char* h_string = (char*)malloc((n + 1) * sizeof(char));
    scanf("%s", h_string);
    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) h_data[i] = h_string[i];    // The character code is the bin

    int *d_data, *d_hist;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_hist, 256 * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    histogramShared<<<blocks, threadsPerBlock>>>(d_data, d_hist, n);

    int h_hist[256];
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Counting sort's second half: emit each letter as often as it was counted
    for (int c = 'a'; c <= 'z'; c++) {
        for (int i = 0; i < h_hist[c]; i++) putchar(c);
    }
    putchar('\n');

    cudaFree(d_data);
    cudaFree(d_hist);
    free(h_string);
    free(h_data);
    return 0;
}
