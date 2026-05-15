#include <stdio.h>
#include <cuda_runtime.h>

__global__ void copyKernel(float* input, float* output, int n) { 
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < n) { 
        output[idx] = input[idx]; 
    } 
}

int main() {
    const int n = 5;
    float h_input[n] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    float h_output[n];
 
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
 
    cudaMemcpy(d_input, h_input, n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0.0, n * sizeof(float));
 
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    copyKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, n);
 
    cudaMemcpy(&h_output, d_output, n * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        printf("%.3f ", h_output[i]);
    }
 
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
