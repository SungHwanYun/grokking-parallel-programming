// Chapter 4 — Practice on swforces
// Problem 78: Multiple Find Element Queries
// https://swforces.com/problem/78
//
// Reference solution (not printed in the book). A map over the queries,
// not over the array: one thread per query scans the whole array for its
// value and writes 1 or 0 to its own output slot. Each output depends on
// the entire input but on no other output, which is what makes it a map.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void findKernel(int* A, int n, int* queries, int* found, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        int k = queries[idx];
        int result = 0;
        for (int i = 0; i < n; i++) {    // Loop over the whole input
            if (A[i] == k) result = 1;
        }
        found[idx] = result;
    }
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    int* h_A = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_A[i]);
    int* h_queries = (int*)malloc(m * sizeof(int));
    for (int i = 0; i < m; i++) scanf("%d", &h_queries[i]);
    int* h_found = (int*)malloc(m * sizeof(int));

    int *d_A, *d_queries, *d_found;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_queries, m * sizeof(int));
    cudaMalloc(&d_found, m * sizeof(int));
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, h_queries, m * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;
    findKernel<<<numBlocks, threadsPerBlock>>>(d_A, n, d_queries, d_found, m);

    cudaMemcpy(h_found, d_found, m * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) printf("%d\n", h_found[i]);

    cudaFree(d_A);
    cudaFree(d_queries);
    cudaFree(d_found);
    free(h_A);
    free(h_queries);
    free(h_found);
    return 0;
}
