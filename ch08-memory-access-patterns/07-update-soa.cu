#include <stdio.h>
#include <cuda_runtime.h>

__global__ void updateSoA(float* x, float* y, float* z,
                          float* vx, float* vy, float* vz,
                          float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += vx[i] * dt;
        y[i] += vy[i] * dt;
        z[i] += vz[i] * dt;
    }
}

int main() {
    const int n = 3;
    float h_x[n]={1.0, 2.0, 3.0};
    float h_y[n]={1.0, 2.0, 3.0};
    float h_z[n]={1.0, 2.0, 3.0};
    float h_vx[n]={1.0, 2.0, 3.0};
    float h_vy[n]={1.0, 2.0, 3.0};
    float h_vz[n]={1.0, 2.0, 3.0};

    float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_z, n * sizeof(float));
    cudaMalloc(&d_vx, n * sizeof(float));
    cudaMalloc(&d_vy, n * sizeof(float));
    cudaMalloc(&d_vz, n * sizeof(float));
 
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, n * sizeof(float), cudaMemcpyHostToDevice);
 
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    updateSoA<<<blocks, threadsPerBlock>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, 2.0, n);
 
    cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, n * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%.3f %.3f %.3f\n", h_x[i], h_y[i], h_z[i]);
    }
 
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_vx);
    cudaFree(d_vy);
    cudaFree(d_vz);
    return 0;
}
