#include <stdio.h>
 
__global__ void average1D(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
 
    float sum = input[idx];
    int count = 1;
 
    if (idx > 0) {
        sum += input[idx - 1];
        count++;
    }
    if (idx < n - 1) {
        sum += input[idx + 1];
        count++;
    }
 
    output[idx] = sum / count;
}

int main() {
    const int n = 5, m = 5;
    float h_input[n] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float h_output[n];
 
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
 
    cudaMemcpy(d_input, h_input, n * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0.0, n * sizeof(float));
 
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    average1D<<<blocks, threadsPerBlock>>>(d_input, d_output, n);
 
    cudaMemcpy(&h_output, d_output, n * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        printf("%.3f ", h_output[i]);
    }
 
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
