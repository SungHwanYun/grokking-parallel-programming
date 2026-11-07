// Chapter 4 — Practice on swforces
// Problem 57: Convert Lowercase to Uppercase
// https://swforces.com/problem/57
//
// Reference solution (not printed in the book). A pure map on characters:
// one thread, one character, and B[idx] depends only on A[idx]. Lowercase
// letters are 32 code points above their uppercase versions, so the
// conversion is a subtraction guarded by a range check.

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void upperKernel(char* A, char* B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        char c = A[idx];
        B[idx] = (c >= 'a' && c <= 'z') ? c - 32 : c;    // 'a' - 32 == 'A'
    }
}

int main() {
    int n;
    char h_A[128], h_B[128];
    scanf("%d", &n);
    scanf("%s", h_A);

    char *d_A, *d_B;
    cudaMalloc(&d_A, n * sizeof(char));
    cudaMalloc(&d_B, n * sizeof(char));
    cudaMemcpy(d_A, h_A, n * sizeof(char), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    upperKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, n);

    cudaMemcpy(h_B, d_B, n * sizeof(char), cudaMemcpyDeviceToHost);
    h_B[n] = '\0';
    printf("%s\n", h_B);

    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
