// Chapter 9 — The Stencil Pattern
// A 1D stencil: 3-point moving average with boundary handling (kernel only).

__global__ void average1D(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float sum = input[idx];    // Start with yourself
    int count = 1;

    if (idx > 0) {    // Left neighbor
        sum += input[idx - 1];
        count++;
    }
    if (idx < n - 1) {    // Right neighbor
        sum += input[idx + 1];
        count++;
    }

    output[idx] = sum / count;
}
