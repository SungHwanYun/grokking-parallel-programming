// Chapter 10 — Memory access patterns
// AoS versus SoA: the two layouts, the updateX kernel that touches two
// contiguous SoA arrays, and the BEFORE/AFTER particle update (kernels only).

#define N 1000000    // not in the book; lets this listing compile standalone

struct Particle {
    float x, y, z;    // position
    float vx, vy, vz;    // velocity
};

Particle particles[N];    // Array of Structures (AoS)

// Structure of Arrays
float x[N], y[N], z[N];
float vx[N], vy[N], vz[N];

__global__ void updateX(float* x, float* vx, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] += vx[idx];    // Both are coalesced
    }
}

// BEFORE: AoS (stride 6: about one byte in six is useful)
__global__ void updateAoS(Particle* p, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx * dt;    // Stride 6
        p[i].y += p[i].vy * dt;    // The same field of consecutive particles is 24 bytes apart
        p[i].z += p[i].vz * dt;
    }
}
// AFTER: SoA (stride 1: coalesced)
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
