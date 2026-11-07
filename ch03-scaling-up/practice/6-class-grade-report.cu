// Chapter 3 — Practice on swforces
// Problem 6: Class Grade Report
// https://swforces.com/problem/6
//
// Reference solution (not printed in the book). One thread per student,
// launched with the chapter's recipe: pick a block size, compute the block
// count with ceiling division, compute the global index, keep the bounds
// check. Each student's six numbers (five scores and the ID) sit together
// in a flat array, so thread i reads scores[6 * i] through scores[6 * i + 4].

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void gradeKernel(int* scores, int* total, int* average, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int* s = scores + 6 * idx;    // This student's row: k m e s c t
        int sum = s[0] + s[1] + s[2] + s[3] + s[4];
        total[idx] = sum;
        average[idx] = sum / 5;
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int* h_scores = (int*)malloc(6 * n * sizeof(int));
    for (int i = 0; i < 6 * n; i++) scanf("%d", &h_scores[i]);
    int* h_total = (int*)malloc(n * sizeof(int));
    int* h_average = (int*)malloc(n * sizeof(int));

    int *d_scores, *d_total, *d_average;
    cudaMalloc(&d_scores, 6 * n * sizeof(int));
    cudaMalloc(&d_total, n * sizeof(int));
    cudaMalloc(&d_average, n * sizeof(int));
    cudaMemcpy(d_scores, h_scores, 6 * n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    gradeKernel<<<numBlocks, threadsPerBlock>>>(d_scores, d_total, d_average, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_total, d_total, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_average, d_average, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) printf("%d %d\n", h_total[i], h_average[i]);

    cudaFree(d_scores);
    cudaFree(d_total);
    cudaFree(d_average);
    free(h_scores);
    free(h_total);
    free(h_average);
    return 0;
}
