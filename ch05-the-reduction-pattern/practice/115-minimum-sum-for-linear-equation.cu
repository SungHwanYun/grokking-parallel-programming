// Chapter 5 — Practice on swforces
// Problem 115: Minimum Sum for Linear Equation
// https://swforces.com/problem/115
//
// Reference solution (not printed in the book). Every solution of
// a*x + y = b is determined by x, and x can only run from 0 to b / a, so
// the candidates are enumerated in parallel: thread x computes
// y = b - a*x and the candidate value x + y. A min reduction (the
// chapter's maxKernel with the operator and identity swapped: INT_MAX and
// atomicMin) picks the winner.

#include <stdio.h>
#include <climits>
#include <cuda_runtime.h>

__global__ void minSumKernel(int a, int b, int* result, int numCandidates) {
    __shared__ int sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int x = idx;
    int y = b - a * x;
    sdata[tid] = (idx < numCandidates) ? x + y : INT_MAX;   // The candidate, or the identity for min
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = min(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin(result, sdata[0]);
    }
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);
    int numCandidates = b / a + 1;    // x = 0, 1, ..., b / a keeps y non-negative
    int threadsPerBlock = 256;
    int numBlocks = (numCandidates + threadsPerBlock - 1) / threadsPerBlock;

    int* d_result;
    cudaMalloc(&d_result, sizeof(int));
    int init = INT_MAX;
    cudaMemcpy(d_result, &init, sizeof(int), cudaMemcpyHostToDevice);

    minSumKernel<<<numBlocks, threadsPerBlock>>>(a, b, d_result, numCandidates);
    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", h_result);
    cudaFree(d_result);
    return 0;
}
