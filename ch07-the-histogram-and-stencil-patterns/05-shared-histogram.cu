#include <stdio.h>
#include <cuda_runtime.h>

__global__ void histogramShared(int* data, int* hist, int n) {
    __shared__ int localHist[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
 
    for (int i = tid; i < 256; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();
 
    if (idx < n) {
        int bin = data[idx];
        atomicAdd(&localHist[bin], 1);
    }
    __syncthreads();
 
    for (int i = tid; i < 256; i += blockDim.x) {
        if (localHist[i] > 0) atomicAdd(&hist[i], localHist[i]);
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
    histogramShared<<<blocks, threadsPerBlock>>>(d_data, d_hist, n);
 
    cudaMemcpy(&h_hist, d_hist, m * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) {
        printf("%d: %d\n", i, h_hist[i]);
    }
 
    cudaFree(d_data);
    cudaFree(d_hist);
    return 0;
}
