// Chapter 10 — Memory Access Patterns
// Matrices: row sums look contiguous but stride; column sums look strided
// but coalesce (kernels only).

// Each thread sums one row, which seems contiguous
__global__ void sumRows(float* matrix, float* result, int width, int height) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    float sum = 0.0f;
    for (int col = 0; col < width; col++) {

        // col=0: T0→[0], T1→[width], T2→[2*width]: Strided!
        sum += matrix[row * width + col];
    }
    result[row] = sum;
}

// Each thread sums one column, which seems strided
__global__ void sumCols(float* matrix, float* result,
                        int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) return;
    float sum = 0.0f;
    for (int row = 0; row < height; row++) {

        sum += matrix[row * width + col];    // row=0: T0→[0], T1→[1], T2→[2]: Coalesced!
    }
    result[col] = sum;
}
