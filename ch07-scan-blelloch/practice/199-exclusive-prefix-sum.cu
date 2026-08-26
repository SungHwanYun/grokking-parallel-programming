// Chapter 7 — Practice on swforces
// Problem 199: Exclusive Prefix Sum
// https://swforces.com/problem/199
//
// Reference solution (not printed in the book). The kernel is the chapter's
// blellochScan unchanged; the problem guarantees n is a power of two, so no
// padding logic is needed (see the chapter's "Power-of-2 Requirement" note).
// Only main() differs, reading the array from standard input.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void blellochScan(int* data, int n) {
    __shared__ int buf[256];
    int tid = threadIdx.x;
    buf[tid] = (tid < n) ? data[tid] : 0;
    __syncthreads();
    // Up-sweep (reduction) phase
    for (int stride = 1; stride < n; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n)
            buf[index] += buf[index - stride];
        __syncthreads();
    }
    // Set last element to 0 (start of exclusive scan)
    if (tid == 0) buf[n - 1] = 0;
    __syncthreads();
    // Down-sweep phase
    for (int stride = n/2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            int t = buf[index];              // Save right child
            buf[index] += buf[index - stride];  // Right = right + left
            buf[index - stride] = t;         // Left = saved old right
        }
        __syncthreads();
    }
    if (tid < n) data[tid] = buf[tid];
}

int main() {
    int n;
    scanf("%d", &n);
    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_data[i]);

    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    blellochScan<<<1, 256>>>(d_data, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) printf("%d ", h_data[i]);
    printf("\n");

    cudaFree(d_data);
    free(h_data);
    return 0;
}
