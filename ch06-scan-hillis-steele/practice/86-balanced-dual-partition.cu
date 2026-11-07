// Chapter 6 — Practice on swforces
// Problem 86: Balanced Dual Partition
// https://swforces.com/problem/86
//
// Reference solution (not printed in the book). With the inclusive prefix
// sums P, a split after position i gives S(X) = P[i] and
// S(Y) = P[n - 1] - P[i], so every split can be judged from one scan. The
// chapter's Hillis-Steele scanKernel computes P; the host then counts the
// splits i = 0 .. n - 2 (both parts non-empty) with S(X) <= S(Y).

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void scanKernel(int* input, int* output, int n) {
    __shared__ int buf[2][256];  // Double buffer
    int tid = threadIdx.x;
    int pout = 0, pin = 1;
    // Load input into shared memory
    buf[pout][tid] = (tid < n) ? input[tid] : 0;
    __syncthreads();
    // Main loop: stride doubles each round
    for (int stride = 1; stride < n; stride *= 2) {
        pout = 1 - pout;  // Swap buffers
        pin  = 1 - pout;
        if (tid >= stride) {
            buf[pout][tid] = buf[pin][tid]
                           + buf[pin][tid - stride];
        } else {
            buf[pout][tid] = buf[pin][tid];
        }
        __syncthreads();
    }
    // Write result to global memory
    if (tid < n) output[tid] = buf[pout][tid];
}

int main() {
    int n;
    scanf("%d", &n);
    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_data[i]);

    int *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMemcpy(d_input, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    scanKernel<<<1, 256>>>(d_input, d_output, n);

    cudaMemcpy(h_data, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    int total = h_data[n - 1];    // The last prefix sum is the sum of the whole array
    int count = 0;
    for (int i = 0; i < n - 1; i++) {
        int sumX = h_data[i];
        int sumY = total - sumX;
        if (sumX <= sumY) count++;
    }
    printf("%d\n", count);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_data);
    return 0;
}
