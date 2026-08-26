// Chapter 11 — Practice on swforces
// Problem 206: Dot Product
// https://swforces.com/problem/206
//
// Reference solution (not printed in the book). The 3-Step Recipe applied
// to a new problem: each thread Loads its own product a[i]*b[i] into shared
// memory, the block Collaborates on the tree reduction from Chapter 5, and
// thread 0 Writes back the block's partial sum with one atomicAdd. n can be
// up to 2,048, so several blocks contribute partial sums.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void dotKernel(int* a, int* b, int* result, int n) {
    __shared__ int sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Load — each thread's own product (or 0 as padding)
    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0;
    __syncthreads();

    // Step 2: Collaborate — tree reduction, one barrier per round
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Step 3: Write back — one atomicAdd per block
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int* h_a = (int*)malloc(n * sizeof(int));
    int* h_b = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_a[i]);
    for (int i = 0; i < n; i++) scanf("%d", &h_b[i]);

    int *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(int));  // Zero the accumulator before launch

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dotKernel<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_result, n);

    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", h_result);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(h_a);
    free(h_b);
    return 0;
}
