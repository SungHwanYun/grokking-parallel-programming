#include <stdio.h>
 
__global__ void histogramAtomic(int* data, int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = data[idx];
        atomicAdd(&hist[bin], 1);
    }
}
 
int main() {
    const int n = 2000, m = 8;
    int h_data[n] = { 0 };
    int h_hist[m] = { 0 };
 
    int *d_data, *d_hist;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_hist, m * sizeof(int));
 
    cudaMemcpy(d_data, h_data, n * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, m * sizeof(int));
 
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    histogramAtomic<<<blocks, threadsPerBlock>>>(d_data, d_hist, n);
 
    cudaMemcpy(&h_hist, d_hist, m * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) {
        printf("%d: %d\n", i, h_hist[i]);
    }
 
    cudaFree(d_data);
    cudaFree(d_hist);
    return 0;
}
