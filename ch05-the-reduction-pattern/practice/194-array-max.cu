// Chapter 5 — Practice on swforces
// Problem 194: Array Max
// https://swforces.com/problem/194
//
// Reference solution. The kernel is the book's maxKernel: swap += for max(),
// swap the identity element 0 for INT_MIN, and swap atomicAdd for atomicMax.
// The accumulator must start at INT_MIN, and cudaMemset sets bytes, not
// ints, so we initialize it with a one-int cudaMemcpy instead.

#include <stdio.h>
#include <stdlib.h>
#include <climits>           // for INT_MIN
#include <cuda_runtime.h>

__global__ void maxKernel(int* data, int* result, int n) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? data[idx] : INT_MIN;   // Identity element for max
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
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_data[i]);
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    int init = INT_MIN;
    cudaMemcpy(d_result, &init, sizeof(int), cudaMemcpyHostToDevice);

    maxKernel<<<numBlocks, threadsPerBlock>>>(d_data, d_result, n);
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", h_result);
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    return 0;
}
