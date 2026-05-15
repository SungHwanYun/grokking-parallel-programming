#include <stdio.h>
#include <cuda_runtime.h>

__global__ void copyScattered(float* input, float* output,
                              int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[indices[idx]];
    }
}

int main() {
    const int n = 10;
    float h_input[n] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    float h_output[n];
    int h_indices[n] = {4, 1, 9, 2, 5, 8, 3, 7, 6, 0};
 
    float *d_input, *d_output;
    int *d_indices;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    cudaMalloc(&d_indices, n * sizeof(float));
 
    cudaMemcpy(d_input, h_input, n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0.0, n * sizeof(float));
 
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    copyScattered<<<blocks, threadsPerBlock>>>(d_input, d_output, d_indices, n);
 
    cudaMemcpy(&h_output, d_output, n * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        printf("%.3f ", h_output[i]);
    }
 
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_indices);
    return 0;
}
