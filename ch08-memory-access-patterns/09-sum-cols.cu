#include <stdio.h>
#include <cuda_runtime.h>

__global__ void sumCols(float* matrix, float* result,
                        int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    float sum = 0.0f;
    for (int row = 0; row < height; row++) {
        sum += matrix[row * width + col];
    }
    result[col] = sum;
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
    cudaMalloc(&d_output, m * sizeof(float));
 
    cudaMemcpy(d_input, h_input, n * m * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0.0, m * sizeof(float));
 
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    sumCols<<<blocks, threadsPerBlock>>>(d_input, d_output, m, n);
 
    cudaMemcpy(&h_output, d_output, m * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) {
        printf("%.3f ", h_output[i]);
    }
 
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
