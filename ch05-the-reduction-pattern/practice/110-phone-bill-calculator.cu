// Chapter 5 — Practice on swforces
// Problem 110: Phone Bill Calculator
// https://swforces.com/problem/110
//
// Reference solution (not printed in the book). The host parses each
// "HH:MM" into minutes (scanf's "%d:%d" does the splitting). On the device,
// each thread turns its minutes into a fee: the base fee, plus one unit fee
// for every started 50 minutes beyond the first 100. The chapter's sum
// reduction adds the fees.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ int fee(int minutes) {
    if (minutes <= 100) return 10;                   // Base fee covers 100 minutes
    int excess = minutes - 100;
    return 10 + ((excess + 49) / 50) * 3;            // Ceiling division: a started unit counts
}

__global__ void billKernel(int* minutes, int* result, int n) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? fee(minutes[idx]) : 0;   // Each thread's own fee (or 0 as padding)
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    int* h_minutes = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        int hh, mm;
        scanf("%d:%d", &hh, &mm);
        h_minutes[i] = hh * 60 + mm;
    }
    int* d_minutes;
    cudaMalloc(&d_minutes, n * sizeof(int));
    cudaMemcpy(d_minutes, h_minutes, n * sizeof(int), cudaMemcpyHostToDevice);

    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));

    billKernel<<<numBlocks, threadsPerBlock>>>(d_minutes, d_result, n);
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", h_result);
    cudaFree(d_minutes);
    cudaFree(d_result);
    free(h_minutes);
    return 0;
}
