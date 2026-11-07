// Chapter 2 — Practice on swforces
// Problem 7: Swap Two Variables
// https://swforces.com/problem/7
//
// Reference solution (not printed in the book). Two threads, two elements:
// thread 0 writes input[1] to output[0] and thread 1 writes input[0] to
// output[1]. Reading from one array and writing to another is what makes
// the swap safe; with a single array, one thread could overwrite the value
// the other thread still has to read.

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void swapKernel(int* input, int* output) {
    int idx = threadIdx.x;
    output[idx] = input[1 - idx];    // Thread 0 takes element 1, thread 1 takes element 0
}

int main() {
    int h_data[2];
    scanf("%d %d", &h_data[0], &h_data[1]);

    int *d_input, *d_output;
    cudaMalloc(&d_input, 2 * sizeof(int));
    cudaMalloc(&d_output, 2 * sizeof(int));
    cudaMemcpy(d_input, h_data, 2 * sizeof(int), cudaMemcpyHostToDevice);

    swapKernel<<<1, 2>>>(d_input, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_output, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d %d\n", h_data[0], h_data[1]);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
