#include <stdio.h>
#include <cuda_runtime.h>    
__global__ void doubleKernel(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    
    if (idx < n) {    
        data[idx] = data[idx] * 2;
    }
}
int main() {
    int n;
    scanf("%d", &n);
    int size = n * sizeof(int);

    int* h_data = (int*)malloc(size);    
    for (int i = 0; i < n; i++) scanf("%d", &h_data[i]);

    int* d_data;
    cudaMalloc(&d_data, size);
    
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;    
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    doubleKernel<<<numBlocks, threadsPerBlock>>>(d_data, n);    
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) printf("%d ", h_data[i]);

    cudaFree(d_data);
    free(h_data);
    return 0;
}
