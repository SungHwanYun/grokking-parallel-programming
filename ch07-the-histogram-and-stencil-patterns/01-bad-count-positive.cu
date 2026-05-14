#include <stdio.h>
 
__global__ void countPositive(int* data, int* count, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && data[idx] > 0) {
        (*count)++;
    }
}
 
int main() {
    int h_data[] = {3, -1, 4, -1, 5, -9, 2, 6};
    int n = 8;
    int hostCount = 0;
 
    int *d_data, *d_count;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
 
    cudaMemcpy(d_data, h_data, n * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &hostCount, sizeof(int),
               cudaMemcpyHostToDevice);
 
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    countPositive<<<blocks, threadsPerBlock>>>(d_data, d_count, n);
 
    cudaMemcpy(&hostCount, d_count, sizeof(int),
               cudaMemcpyDeviceToHost);
    printf("Positive count: %d\n", hostCount);
 
    cudaFree(d_data);
    cudaFree(d_count);
    return 0;
}
