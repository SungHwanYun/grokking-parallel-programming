#include <stdio.h>
#include <cuda_runtime.h>

struct Particle {
    float x, y, z;
    float vx, vy, vz;
};

__global__ void updateXAoS(Particle* p, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        p[idx].x += p[idx].vx;
    }
}

int main() {
    const int n = 3;
    Particle h_particles[n] = {
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0, 2.0, 2.0, 2.0},
        {3.0, 3.0, 3.0, 3.0, 3.0, 3.0}
    };

    Particle *d_particles;
    cudaMalloc(&d_particles, n * sizeof(Particle));
 
    cudaMemcpy(d_particles, h_particles, n * sizeof(Particle),
               cudaMemcpyHostToDevice);
 
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    updateXAoS<<<blocks, threadsPerBlock>>>(d_particles, n);
 
    cudaMemcpy(&h_particles, d_particles, n * sizeof(Particle),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        printf("%.3f ", h_particles[i].x);
    }
 
    cudaFree(d_particles);
    return 0;
}
