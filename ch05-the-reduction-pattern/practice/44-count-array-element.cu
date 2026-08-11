// Chapter 5 — Practice on swforces
// Problem 44: Count Array Element
// https://swforces.com/problem/44
//
// Reference solution. Counting is a sum reduction in disguise: each thread
// loads 1 if its element equals k and 0 otherwise, then the tree reduction
// and atomicAdd from the chapter's sum program do the rest unchanged.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void countKernel(int* data, int* result, int n, int k) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n && data[idx] == k) ? 1 : 0;   // 1 if it matches, else 0
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
    int n, k;
    scanf("%d %d", &n, &k);
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

    countKernel<<<numBlocks, threadsPerBlock>>>(d_data, d_result, n, k);
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", h_result);
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    return 0;
}
