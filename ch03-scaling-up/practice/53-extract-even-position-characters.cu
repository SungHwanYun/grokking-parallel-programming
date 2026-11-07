// Chapter 3 — Practice on swforces
// Problem 53: Extract Even-Position Characters
// https://swforces.com/problem/53
//
// Reference solution (not printed in the book). The even positions
// (2nd, 4th, ...) are the odd 0-based indices 1, 3, 5, ..., so the output
// has n / 2 characters and output[idx] = A[2 * idx + 1]. One thread per
// output character, launched with the chapter's recipe.

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void evenPositionKernel(char* A, char* B, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        B[idx] = A[2 * idx + 1];    // Position 2(idx + 1) in 1-based terms
    }
}

int main() {
    int n;
    char h_A[128], h_B[128];
    scanf("%d", &n);
    scanf("%s", h_A);
    int m = n / 2;    // Number of even positions

    char *d_A, *d_B;
    cudaMalloc(&d_A, n * sizeof(char));
    cudaMalloc(&d_B, m * sizeof(char));
    cudaMemcpy(d_A, h_A, n * sizeof(char), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;
    evenPositionKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, m);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, m * sizeof(char), cudaMemcpyDeviceToHost);
    h_B[m] = '\0';
    printf("%s\n", h_B);

    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}
