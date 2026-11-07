// Chapter 2 — Practice on swforces
// Problem 12: Sort Three Numbers
// https://swforces.com/problem/12
//
// Reference solution (not printed in the book). A rank sort: each of the
// three threads counts how many elements are smaller than its own, and
// that count is the position where its element belongs in the sorted
// output. Ties are broken by index (an equal element with a smaller index
// counts as smaller), so equal values never claim the same slot. Every
// thread only reads input and writes its own slot of output, so there is
// no conflict.

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void rankSortKernel(int* input, int* output, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        int rank = 0;
        for (int i = 0; i < n; i++) {
            if (input[i] < input[idx] || (input[i] == input[idx] && i < idx)) {
                rank++;
            }
        }
        output[rank] = input[idx];
    }
}

int main() {
    int n = 3;
    int h_data[3];
    scanf("%d %d %d", &h_data[0], &h_data[1], &h_data[2]);

    int *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMemcpy(d_input, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    rankSortKernel<<<1, n>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d %d %d\n", h_data[0], h_data[1], h_data[2]);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
