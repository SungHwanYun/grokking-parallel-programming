// Chapter 11 — Practice on swforces
// Problem 67: Max Subarray Sum
// https://swforces.com/problem/67
//
// Reference solution (not printed in the book). Neighboring windows
// overlap in w - 1 elements, so the array is read many times over: a job
// for the 3-Step Recipe. Load: the block copies the whole array (n <= 100)
// into shared memory once. Collaborate: thread i sums window i from shared
// memory, then the block runs chapter 5's tree reduction with max over the
// window sums. Write back: thread 0 writes the single result. One block of
// 128 threads covers every window, so no atomic merge is needed.

#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128    // A power of 2 that is at least n

__global__ void maxWindowKernel(int* data, int* result, int n, int w) {
    __shared__ int sdata[BLOCK_SIZE];    // The array, loaded once
    __shared__ int sums[BLOCK_SIZE];     // One window sum per thread
    int tid = threadIdx.x;

    // Step 1: Load. Everyone copies one element from Global to Shared
    sdata[tid] = (tid < n) ? data[tid] : 0;
    __syncthreads();

    // Step 2: Collaborate. Each window sum reads w elements from the whiteboard, not from global memory
    int numWindows = n - w + 1;
    int sum = INT_MIN;    // Identity for max, for threads without a window
    if (tid < numWindows) {
        sum = 0;
        for (int i = 0; i < w; i++) sum += sdata[tid + i];
    }
    sums[tid] = sum;
    __syncthreads();

    // ... then the tree reduction from chapter 5, with max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] = max(sums[tid], sums[tid + stride]);
        }
        __syncthreads();
    }

    // Step 3: Write back. One block, so thread 0 holds the final answer
    if (tid == 0) *result = sums[0];
}

int main() {
    int n, w;
    scanf("%d %d", &n, &w);
    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_data[i]);

    int *d_data, *d_result;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    maxWindowKernel<<<1, BLOCK_SIZE>>>(d_data, d_result, n, w);

    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", h_result);

    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    return 0;
}
