// Chapter 9 — Practice on swforces
// Problem 203: 1D Moving Average
// https://swforces.com/problem/203
//
// Reference solution (not printed in the book). The kernel is the chapter's
// average1D unchanged: start with yourself, add the neighbors that exist,
// divide by the count. main() reads the array, launches one thread per
// element, and writes to a separate output array (never in place).

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void average1D(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float sum = input[idx];    // Start with yourself
    int count = 1;

    if (idx > 0) {    // Left neighbor
        sum += input[idx - 1];
        count++;
    }
    if (idx < n - 1) {    // Right neighbor
        sum += input[idx + 1];
        count++;
    }

    output[idx] = sum / count;
}

int main() {
    int n;
    scanf("%d", &n);
    float* h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) scanf("%f", &h_data[i]);

    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    cudaMemcpy(d_input, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    average1D<<<blocks, threadsPerBlock>>>(d_input, d_output, n);

    cudaMemcpy(h_data, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) printf("%.6f ", h_data[i]);
    printf("\n");

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_data);
    return 0;
}
