// Chapter 2 — Practice on swforces
// Problem 33: Greater Than
// https://swforces.com/problem/33
//
// Reference solution (not printed in the book). One thread evaluates the
// comparison; the result (1 or 0) is written to device memory and copied
// back, exactly like the sum in Problem 21.

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void greaterKernel(int a, int b, int* result) {
    *result = (a > b) ? 1 : 0;
}

int main() {
    int a, b;
    scanf("%d %d", &a, &b);

    int* d_result;
    cudaMalloc(&d_result, sizeof(int));

    greaterKernel<<<1, 1>>>(a, b, d_result);
    cudaDeviceSynchronize();

    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", h_result);

    cudaFree(d_result);
    return 0;
}
