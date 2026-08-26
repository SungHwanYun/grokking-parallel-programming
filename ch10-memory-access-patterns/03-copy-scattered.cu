// Chapter 10 — Memory Access Patterns
// Scattered access: the worst case, indices jump around randomly (kernel only).

__global__ void copyScattered(float* input, float* output,
                              int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[indices[idx]];    // Random!
    }
}
