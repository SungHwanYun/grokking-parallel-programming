#include <stdio.h>
 
__global__ void blur2D(float* input, float* output,
                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
 
    float sum = 0.0f;
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width &&
                ny >= 0 && ny < height) {
                sum += input[ny * width + nx];
                count++;
            }
        }
    }
    output[y * width + x] = sum / count;
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
    float h_output[n][m];
 
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * m * sizeof(float));
    cudaMalloc(&d_output, n * m * sizeof(float));
 
    cudaMemcpy(d_input, h_input, n * m * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0.0, n * m * sizeof(float));
 
    dim3 threadsPerBlock(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    blur2D<<<blocks, threadsPerBlock>>>(d_input, d_output, n, m);
 
    cudaMemcpy(&h_output, d_output, n * m * sizeof(int),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%.3f ", h_output[i][j]);
        }
        printf("\n");
    }
 
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
