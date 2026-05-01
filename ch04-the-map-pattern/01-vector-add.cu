#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAddKernel(int *A, int *B, int *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int n;
    scanf("%d", &n);
    
    int *h_A = (int*)malloc(n * sizeof(int));
    int *h_B = (int*)malloc(n * sizeof(int));
    int *h_C = (int*)malloc(n * sizeof(int));
    
    for (int i = 0; i < n; i++) scanf("%d", &h_A[i]);
    for (int i = 0; i < n; i++) scanf("%d", &h_B[i]);
    
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_C, n * sizeof(int));
    
    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(int), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    
    cudaMemcpy(h_C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n; i++) printf("%d ", h_C[i]);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    
    return 0;
}
