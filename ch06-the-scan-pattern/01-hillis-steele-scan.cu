#include <stdio.h>
#include <cuda_runtime.h>
__global__ void hillisSteeleScanKernel(int* input, int* output, int n) {
    __shared__ int buf[2][256];  // Double buffer
    int tid = threadIdx.x;
    int pout = 0, pin = 1;
    // Load input into shared memory
    buf[pout][tid] = (tid < n) ? input[tid] : 0;
    __syncthreads();
    // Main loop: stride doubles each round
    for (int stride = 1; stride < n; stride *= 2) {
        pout = 1 - pout;  // Swap buffers
        pin  = 1 - pout;
        if (tid >= stride) {
            buf[pout][tid] = buf[pin][tid]
                           + buf[pin][tid - stride];
        } else {
            buf[pout][tid] = buf[pin][tid];
        }
        __syncthreads();
    }
    // Write result to global memory
    if (tid < n) output[tid] = buf[pout][tid];
}
int main() {
    int n = 8;
    int h_input[] = {3, 1, 7, 0, 4, 1, 6, 3};
    int h_output[8];
    int *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
    hillisSteeleScanKernel<<<1, 256>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Scan result: ");
    for (int i = 0; i < n; i++) printf("%d ", h_output[i]);
    printf("\n");
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
