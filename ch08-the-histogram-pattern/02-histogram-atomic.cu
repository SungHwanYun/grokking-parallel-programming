// Chapter 8 — The Histogram Pattern
// A first fix: 256-bin histogram with global atomics (kernel only).

__global__ void histogramAtomic(int* data, int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = data[idx];
        atomicAdd(&hist[bin], 1);    // Safe!
    }
}
