// Chapter 10 — Memory Access Patterns
// AoS versus SoA: the same particle update in both layouts (kernels only).

#define N 1000000    // not in the book; lets this listing compile standalone

struct Particle {
    float x, y, z;    // position
    float vx, vy, vz;    // velocity
};

Particle particles[N];    // Array of Structures (AoS)

// BEFORE: AoS (stride of 6, ~17% bandwidth efficiency)
__global__ void updateAoS(Particle* p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx * dt;    // Stride 6
        p[i].y += p[i].vy * dt;    // Each field is 24 bytes apart
        p[i].z += p[i].vz * dt;
    }
}
// AFTER: SoA (stride of 1, ~70% of peak bandwidth in practice)
__global__ void updateSoA(float* x, float* y, float* z,
                          float* vx, float* vy, float* vz,
                          float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += vx[i] * dt;    // Stride 1
        y[i] += vy[i] * dt;    // Each array is contiguous
        z[i] += vz[i] * dt;
    }
}
