// Chapter 2 — Practice on swforces
// Problem 21: Sum of Two Integers
// https://swforces.com/problem/21
//
// Reference solution (not printed in the book). The smallest possible
// five-step program: a and b travel as kernel arguments (like n in the
// chapter's squareKernel), one thread adds them, and the single result comes
// back through cudaMemcpy.

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void addKernel(int a, int b, int* result) {
    *result = a + b;
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);

    int* d_result;
    cudaMalloc(&d_result, sizeof(int));

    addKernel<<<1, 1>>>(a, b, d_result);
    cudaDeviceSynchronize();

    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", h_result);

    cudaFree(d_result);
    return 0;
}
