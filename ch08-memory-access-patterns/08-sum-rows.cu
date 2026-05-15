#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sumRows(float* matrix, float* result, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    float sum = 0.0f;
    for (int col = 0; col < width; col++) {
        sum += matrix[row * width + col];
    }
    result[row] = sum;
}

int main() {
    const int n = 5, m = 5;
    float h_input[n][m] = { 
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {1.0, 2.0, 3.0, 4.0, 5.0},  
        {1.0, 2.0, 3.0, 4.0, 5.0}
    };
    float h_output[n];
 
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * m * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
 
    cudaMemcpy(d_input, h_input, n * m * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0.0, n * sizeof(float));
 
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    sumRows<<<blocks, threadsPerBlock>>>(d_input, d_output, m, n);
 
    cudaMemcpy(&h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        printf("%.3f ", h_output[i]);
    }
 
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
