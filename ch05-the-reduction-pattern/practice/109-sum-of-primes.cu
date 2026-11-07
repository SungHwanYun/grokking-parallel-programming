// Chapter 5 — Practice on swforces
// Problem 109: Sum of Primes
// https://swforces.com/problem/109
//
// Reference solution (not printed in the book). Each thread first turns
// its element into a value (the element itself if it is prime, else 0),
// using a trial-division helper marked __device__ so the kernel can call
// it. The chapter's sum reduction then adds those values.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ int isPrime(int x) {
    if (x < 2) return 0;
    for (int d = 2; d * d <= x; d++) {
        if (x % d == 0) return 0;
    }
    return 1;
}

__global__ void primeSumKernel(int* data, int* result, int n) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n && isPrime(data[idx])) ? data[idx] : 0;   // A prime or 0
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

    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_data[i]);
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    cudaMemset(d_result, 0, sizeof(int));

    primeSumKernel<<<numBlocks, threadsPerBlock>>>(d_data, d_result, n);
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", h_result);
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    return 0;
}
