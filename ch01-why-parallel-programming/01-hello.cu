#include <stdio.h>
#include <cuda_runtime.h>
 
__global__ void helloKernel() {
    printf("Hello, Parallel World!\n");
}

int main() {
    helloKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
