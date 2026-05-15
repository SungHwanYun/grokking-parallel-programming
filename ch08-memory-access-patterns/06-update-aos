#include <stdio.h>
#include <cuda_runtime.h>

struct Particle {
    float x, y, z;
    float vx, vy, vz;
};

__global__ void updateAoS(Particle* p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
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
    updateAoS<<<blocks, threadsPerBlock>>>(d_particles, 2.0, n);
 
    cudaMemcpy(&h_particles, d_particles, n * sizeof(Particle),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        printf("%.3f %.3f %.3f\n", h_particles[i].x, h_particles[i].y, h_particles[i].z);
    }
 
    cudaFree(d_particles);
    return 0;
}
