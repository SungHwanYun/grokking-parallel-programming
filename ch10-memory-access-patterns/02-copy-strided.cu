// Chapter 10 — Memory Access Patterns
// Strided access: each thread skips elements (kernel only).

__global__ void copyStrided(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx * 2];  // Stride of 2!
    }
}
