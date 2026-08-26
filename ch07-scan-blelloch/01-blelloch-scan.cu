// Chapter 7 — Work-Efficient Scan: Blelloch
// Complete Blelloch exclusive scan (single block; n must be a power of 2).
// Up-sweep builds the reduction tree; the down-sweep distributes prefix sums.

#include <stdio.h>
#include <cuda_runtime.h>
__global__ void blellochScan(int* data, int n) {
    __shared__ int buf[256];
    int tid = threadIdx.x;
    buf[tid] = (tid < n) ? data[tid] : 0;
    __syncthreads();    // Part 1: Load data into shared memory
    for (int stride = 1; stride < n; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n)
            buf[index] += buf[index - stride];
        __syncthreads();    // Part 2: Up-sweep (reduction) phase
    }
    if (tid == 0) buf[n - 1] = 0;
    __syncthreads();    // Part 3: Set root to identity element
    for (int stride = n/2; stride > 0; stride /= 2) {    // Part 4: Down-sweep phase
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            int t = buf[index];    // Save right child
            buf[index] += buf[index - stride];    // Right = right + left
            buf[index - stride] = t;    // Left = saved old right
        }
        __syncthreads();
    }
    if (tid < n) data[tid] = buf[tid];    // Part 5: Write result back
}
int main() {
    int n = 8;
    int h_data[] = {3, 1, 7, 0, 4, 1, 6, 3};
    int *d_data;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int),
        cudaMemcpyHostToDevice);
    blellochScan<<<1, 256>>>(d_data, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, n * sizeof(int),
        cudaMemcpyDeviceToHost);
    printf("Exclusive scan: ");
    for (int i = 0; i < n; i++) printf("%d ", h_data[i]);
    printf("\n");
    cudaFree(d_data);
    return 0;
}
