// Chapter 10 — Memory Access Patterns
// Coalesced access: consecutive threads read consecutive addresses (kernel only).

__global__ void copyKernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}
