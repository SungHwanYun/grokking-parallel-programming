#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printIndexKernel() {
    int idx = threadIdx.x;
    if (idx == 2)
        printf("Thread %d: Hello!\n", idx);
}

int main() {
    printIndexKernel<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
