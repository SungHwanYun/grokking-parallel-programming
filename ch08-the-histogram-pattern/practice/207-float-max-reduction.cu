// Chapter 8 — Practice on swforces
// Problem 207: Float Max Reduction
// https://swforces.com/problem/207
//
// Reference solution (not printed in the book). Chapter 5's max reduction
// on floats. The tree reduction uses fmaxf and the identity -FLT_MAX, and
// the merge across blocks uses the sidebar's atomicMaxFloat: atomicMax
// exists only for integers, so the float's bits are reinterpreted as an
// unsigned int and swapped in with atomicCAS, retrying until no other
// thread has written a larger value in between. The result starts at
// -FLT_MAX (not 0: the array may be entirely negative), copied in with
// cudaMemcpy because cudaMemset can only fill bytes.

#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <cuda_runtime.h>

__device__ float atomicMaxFloat(float* addr, float val) {
    unsigned int* p = (unsigned int*)addr;
    unsigned int old = *p, assumed;
    do {
        assumed = old;
        if (__uint_as_float(assumed) >= val) break;    // Already at least as large: nothing to do
        old = atomicCAS(p, assumed, __float_as_uint(val));    // Swap in our bits only if nobody else wrote first
    } while (assumed != old);    // Someone else wrote first: compare against their value
    return __uint_as_float(old);
}

__global__ void maxKernel(float* data, float* result, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? data[idx] : -FLT_MAX;    // Identity element for max
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMaxFloat(result, sdata[0]);
    }
}

int main() {
    int n;
    scanf("%d", &n);
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    float* h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) scanf("%f", &h_data[i]);
    float* d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    float init = -FLT_MAX;
    cudaMemcpy(d_result, &init, sizeof(float), cudaMemcpyHostToDevice);

    maxKernel<<<numBlocks, threadsPerBlock>>>(d_data, d_result, n);
    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%.6f\n", h_result);
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    return 0;
}
