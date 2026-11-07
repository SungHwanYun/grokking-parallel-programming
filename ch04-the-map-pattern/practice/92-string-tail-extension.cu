// Chapter 4 — Practice on swforces
// Problem 92: String Tail Extension
// https://swforces.com/problem/92
//
// Reference solution (not printed in the book). Appending the last
// character over and over is a sequential description, but the result is a
// map: output position idx holds A[idx] if idx is inside the original
// string and A[len - 1] otherwise. One thread per output character, k of
// them in all; every thread past the original length reads the same last
// character.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void extendKernel(char* A, char* B, int len, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        B[idx] = (idx < len) ? A[idx] : A[len - 1];
    }
}

int main() {
    char h_A[128];
    int k;
    scanf("%s", h_A);
    scanf("%d", &k);
    int len = strlen(h_A);
    char* h_B = (char*)malloc((k + 1) * sizeof(char));

    char *d_A, *d_B;
    cudaMalloc(&d_A, len * sizeof(char));
    cudaMalloc(&d_B, k * sizeof(char));
    cudaMemcpy(d_A, h_A, len * sizeof(char), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (k + threadsPerBlock - 1) / threadsPerBlock;
    extendKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, len, k);

    cudaMemcpy(h_B, d_B, k * sizeof(char), cudaMemcpyDeviceToHost);
    h_B[k] = '\0';
    printf("%s\n", h_B);

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_B);
    return 0;
}
