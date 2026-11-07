// Chapter 10 — Memory access patterns
// Strided access: each thread skips elements (kernel only).

__global__ void copyStrided(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx * 2];    // Stride 2; input must hold at least 2n elements, which the idx < n check does not guarantee
    }
}
