// Chapter 5 — The Reduction pattern
// Problem 5.1: 1D Reduction
// https://swforces.com/problem/49
//
// The kernel has three acts: load onto shared memory (the whiteboard),
// tree reduction (the tournament), report the block result to global memory.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void sumKernel(int* data, int* result, int n) {
    __shared__ int sdata[256];    // The whiteboard, sized to match blockDim.x = 256
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Act 1 — Load: each thread brings one element (or 0 as padding)
    sdata[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads();              // "Pens down!" Wait until everyone has loaded.

    // Act 2 — Tree reduction: halve the active threads each round
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();          // Barrier before the next halving
    }

    // Act 3 — Report: one thread per block adds its partial sum atomically
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;  // Ceiling division

    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_data[i]);
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));   // Must zero the accumulator before launch

    sumKernel<<<numBlocks, threadsPerBlock>>>(d_data, d_result, n);
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);  // Implicitly synchronizes
    printf("%d\n", h_result);
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    return 0;
}
