// Chapter 8 — Practice on swforces
// Problem 87: Favorite Subject Survey
// https://swforces.com/problem/87
//
// Reference solution (not printed in the book). A three-bin histogram.
// The host maps each subject name to a bin (kor = 0, eng = 1, math = 2),
// the chapter's histogramAtomic kernel counts the answers with atomicAdd,
// and each query reads one bin, or all three added together for "-".

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void histogramAtomic(int* data, int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = data[idx];
        atomicAdd(&hist[bin], 1);
    }
}

int subjectBin(const char* name) {
    if (strcmp(name, "kor") == 0) return 0;
    if (strcmp(name, "eng") == 0) return 1;
    if (strcmp(name, "math") == 0) return 2;
    return -1;    // "-": every subject
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    int h_data[16];
    char name[16];
    for (int i = 0; i < n; i++) {
        scanf("%s", name);
        h_data[i] = subjectBin(name);
    }

    int *d_data, *d_hist;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_hist, 3 * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 3 * sizeof(int));    // Bins start at zero

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    histogramAtomic<<<blocks, threadsPerBlock>>>(d_data, d_hist, n);

    int h_hist[3];
    cudaMemcpy(h_hist, d_hist, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int q = 0; q < m; q++) {
        scanf("%s", name);
        int bin = subjectBin(name);
        if (bin >= 0) printf("%d\n", h_hist[bin]);
        else          printf("%d\n", h_hist[0] + h_hist[1] + h_hist[2]);
    }

    cudaFree(d_data);
    cudaFree(d_hist);
    return 0;
}
