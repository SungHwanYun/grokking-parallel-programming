// Chapter 6 — Practice on swforces
// Problem 140: Prime Time Threshold
// https://swforces.com/problem/140
//
// Reference solution (not printed in the book). The value of a at time t
// is 1 plus the sum of all primes up to t, which is an inclusive prefix
// sum over the sequence v[t] = (t if t is prime, else 0). The primes up to
// 97 already add to more than 1,000, so a 256-element scan (t = 1 .. 256,
// one thread per time step) covers every b. The host then finds the first
// t with a >= b.

#include <stdio.h>
#include <cuda_runtime.h>

#define T 256    // Time steps covered by the single block

__device__ int isPrime(int x) {
    if (x < 2) return 0;
    for (int d = 2; d * d <= x; d++) {
        if (x % d == 0) return 0;
    }
    return 1;
}

__global__ void primeScanKernel(int* output) {
    __shared__ int buf[2][T];  // Double buffer
    int tid = threadIdx.x;
    int t = tid + 1;           // Thread tid stands for time t = tid + 1
    int pout = 0, pin = 1;
    // Each thread contributes its own time step: t if prime, else 0
    buf[pout][tid] = isPrime(t) ? t : 0;
    __syncthreads();
    // Main loop: stride doubles each round
    for (int stride = 1; stride < T; stride *= 2) {
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
    // a at time t = 1 + (sum of primes up to t)
    output[tid] = 1 + buf[pout][tid];
}

int main() {
    int b;
    scanf("%d", &b);

    int h_a[T];
    int* d_a;
    cudaMalloc(&d_a, T * sizeof(int));

    primeScanKernel<<<1, T>>>(d_a);

    cudaMemcpy(h_a, d_a, T * sizeof(int), cudaMemcpyDeviceToHost);
    for (int tid = 0; tid < T; tid++) {
        if (h_a[tid] >= b) {
            printf("%d\n", tid + 1);    // The first time t = tid + 1 with a >= b
            break;
        }
    }

    cudaFree(d_a);
    return 0;
}
