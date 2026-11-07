// Chapter 6 — The Scan pattern: Hillis-Steele
// Complete Hillis-Steele inclusive scan (single block, up to 256 elements).
// The double buffer is the only new trick on top of chapter 5's reduction.

#include <stdio.h>
#include <cuda_runtime.h>
__global__ void scanKernel(int* input, int* output, int n) {
    __shared__ int buf[2][256];  // Double buffer
    int tid = threadIdx.x;
    int pout = 0, pin = 1;    // Part 1: Setup with shared memory and the double buffer
    // Load input into shared memory
    buf[pout][tid] = (tid < n) ? input[tid] : 0;    // Part 2: Load input into shared memory
    __syncthreads();
    // Main loop: stride doubles each round
    for (int stride = 1; stride < n; stride *= 2) {    // Part 3: The main loop, doubling stride
        pout = 1 - pout;  // Swap buffers
        pin  = 1 - pout;
        if (tid >= stride) {    // Part 4: The parallel add with boundary check
            buf[pout][tid] = buf[pin][tid]
                           + buf[pin][tid - stride];
        } else {
            buf[pout][tid] = buf[pin][tid];
        }
        __syncthreads();
    }
    // Write result to global memory
    if (tid < n) output[tid] = buf[pout][tid];    // Part 5: Write the result to global memory
}
int main() {
    int n = 8;
    int h_input[] = {3, 1, 7, 0, 4, 1, 6, 3};
    int h_output[8];
    int *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
    scanKernel<<<1, 256>>>(d_input, d_output, n);
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Scan result: ");
    for (int i = 0; i < n; i++) printf("%d ", h_output[i]);
    printf("\n");
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
