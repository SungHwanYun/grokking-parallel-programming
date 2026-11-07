// Chapter 2 — Practice on swforces
// Problem 18: Echo String
// https://swforces.com/problem/18
//
// Reference solution (not printed in the book). The five-step pattern on a
// char array instead of an int array: one thread per character copies its
// character from the input string to the output string, and the host prints
// the result. (Having each thread printf its own character would also pass
// on swforces, but a real GPU gives no guarantee about the printing order.)

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void copyKernel(char* input, char* output, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

int main() {
    char h_input[128];
    scanf("%s", h_input);
    int n = strlen(h_input);
    char h_output[128];

    char *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(char));
    cudaMalloc(&d_output, n * sizeof(char));
    cudaMemcpy(d_input, h_input, n * sizeof(char), cudaMemcpyHostToDevice);

    copyKernel<<<1, n>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, n * sizeof(char), cudaMemcpyDeviceToHost);
    h_output[n] = '\0';    // The kernel copied n characters; terminate the string
    printf("%s\n", h_output);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
