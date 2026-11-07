// Chapter 10 — Memory access patterns
// Scattered access: the worst case, indices jump around randomly (kernel only).

__global__ void copyScattered(float* input, float* output,
                              int* indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[indices[idx]];    // Random; every indices[idx] must be a valid index into input, which the idx < n check does not cover
    }
}
