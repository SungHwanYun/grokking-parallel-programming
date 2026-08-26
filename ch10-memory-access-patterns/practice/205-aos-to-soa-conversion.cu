// Chapter 10 — Practice on swforces
// Problem 205: AoS to SoA Conversion
// https://swforces.com/problem/205
//
// Reference solution (not printed in the book). One thread per particle:
// each thread reads its particle's three interleaved fields (a strided
// access) and writes them to the three separate arrays (which later kernels
// can then read coalesced). This is the layout transformation behind the
// chapter's updateAoS-to-updateSoA rewrite.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void aosToSoA(int* aos, int* x, int* y, int* z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = aos[3 * i];        // Stride 3 on the read side...
        y[i] = aos[3 * i + 1];
        z[i] = aos[3 * i + 2];
    }                             // ...stride 1 on the write side
}

int main() {
    int n;
    scanf("%d", &n);
    int* h_aos = (int*)malloc(3 * n * sizeof(int));
    for (int i = 0; i < 3 * n; i++) scanf("%d", &h_aos[i]);

    int *d_aos, *d_x, *d_y, *d_z;
    cudaMalloc(&d_aos, 3 * n * sizeof(int));
    cudaMalloc(&d_x, n * sizeof(int));
    cudaMalloc(&d_y, n * sizeof(int));
    cudaMalloc(&d_z, n * sizeof(int));
    cudaMemcpy(d_aos, h_aos, 3 * n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    aosToSoA<<<blocks, threadsPerBlock>>>(d_aos, d_x, d_y, d_z, n);

    int* h_out = (int*)malloc(n * sizeof(int));
    int* fields[3] = {d_x, d_y, d_z};
    for (int f = 0; f < 3; f++) {
        cudaMemcpy(h_out, fields[f], n * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) printf("%d ", h_out[i]);
        printf("\n");
    }

    cudaFree(d_aos);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    free(h_aos);
    free(h_out);
    return 0;
}
