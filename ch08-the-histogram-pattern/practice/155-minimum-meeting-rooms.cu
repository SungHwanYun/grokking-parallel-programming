// Chapter 8 — Practice on swforces
// Problem 155: Minimum Meeting Rooms
// https://swforces.com/problem/155
//
// Reference solution (not printed in the book). The rooms needed equal the
// largest number of meetings running at the same moment. Count that with a
// histogram over time: one thread per meeting adds 1 to every time slot
// [start, end) it occupies (the end slot is left out, so a meeting that
// ends at t and one that starts at t share a room). Several meetings hit
// the same slot, so the increments are atomicAdd. The chapter 5 max
// reduction then finds the busiest slot.

#include <stdio.h>
#include <climits>
#include <cuda_runtime.h>

#define SLOTS 101    // Times run from 1 to 100

__global__ void occupancyKernel(int* start, int* end, int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int t = start[idx]; t < end[idx]; t++) {
            atomicAdd(&hist[t], 1);    // This meeting occupies slot t
        }
    }
}

__global__ void maxKernel(int* data, int* result, int n) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? data[idx] : INT_MIN;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(result, sdata[0]);
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int h_start[16], h_end[16];
    for (int i = 0; i < n; i++) scanf("%d %d", &h_start[i], &h_end[i]);

    int *d_start, *d_end, *d_hist, *d_result;
    cudaMalloc(&d_start, n * sizeof(int));
    cudaMalloc(&d_end, n * sizeof(int));
    cudaMalloc(&d_hist, SLOTS * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_start, h_start, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, h_end, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, SLOTS * sizeof(int));
    int init = INT_MIN;
    cudaMemcpy(d_result, &init, sizeof(int), cudaMemcpyHostToDevice);

    occupancyKernel<<<1, 256>>>(d_start, d_end, d_hist, n);
    maxKernel<<<1, 256>>>(d_hist, d_result, SLOTS);

    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", h_result);

    cudaFree(d_start);
    cudaFree(d_end);
    cudaFree(d_hist);
    cudaFree(d_result);
    return 0;
}
