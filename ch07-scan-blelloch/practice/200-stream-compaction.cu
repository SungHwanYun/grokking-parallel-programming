// Chapter 7 — Practice on swforces
// Problem 200: Stream Compaction
// https://swforces.com/problem/200
//
// Reference solution (not printed in the book). This is the chapter's
// "Stream compaction" section made concrete: mark each even element with a
// 0/1 flag, run the chapter's Blelloch exclusive scan over the flags so each
// keeper learns "how many keepers come before me", then scatter. Because n
// can be any size up to 256, the flags are padded to the next power of two
// exactly as the chapter's "Power-of-2 Requirement" note describes; the
// padding flags are 0, so they never claim an output slot.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void compactEven(int* data, int* output, int* count,
                            int n, int padded_n) {
    __shared__ int buf[256];
    int tid = threadIdx.x;

    // Step 1: Predicate — flag the elements to keep (evens)
    int flag = (tid < n && data[tid] % 2 == 0) ? 1 : 0;
    buf[tid] = (tid < padded_n) ? flag : 0;
    __syncthreads();

    // Step 2: Exclusive scan of the flags (Blelloch, as in the chapter)
    for (int stride = 1; stride < padded_n; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < padded_n)
            buf[index] += buf[index - stride];
        __syncthreads();
    }
    if (tid == 0) buf[padded_n - 1] = 0;
    __syncthreads();
    for (int stride = padded_n/2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < padded_n) {
            int t = buf[index];
            buf[index] += buf[index - stride];
            buf[index - stride] = t;
        }
        __syncthreads();
    }

    // Step 3: Scatter — each keeper's destination is its exclusive scan value
    if (tid < n && flag) output[buf[tid]] = data[tid];
    // The last exclusive scan value plus the last flag is the kept count
    if (tid == n - 1) *count = buf[tid] + flag;
}

int main() {
    int n;
    scanf("%d", &n);
    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_data[i]);

    // Pad to the next power of 2 (from the chapter's Power-of-2 note)
    int padded_n = 1;
    while (padded_n < n) padded_n <<= 1;

    int *d_data, *d_output, *d_count;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    compactEven<<<1, 256>>>(d_data, d_output, d_count, n, padded_n);
    cudaDeviceSynchronize();

    int m;
    cudaMemcpy(&m, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data, d_output, m * sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d\n", m);
    for (int i = 0; i < m; i++) printf("%d ", h_data[i]);
    printf("\n");

    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_count);
    free(h_data);
    return 0;
}
