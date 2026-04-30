#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void squareKernel(int* data, int n) {
    int idx = threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}
 
int main() {
    int n;
    scanf("%d", &n);
    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
        scanf("%d", &h_data[i]);
 
    int* d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);
 
    squareKernel<<<1, n>>>(d_data, n);
    cudaDeviceSynchronize();
 
    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        printf("%d ", h_data[i]);
 
    cudaFree(d_data);
    free(h_data);
    return 0;
}
