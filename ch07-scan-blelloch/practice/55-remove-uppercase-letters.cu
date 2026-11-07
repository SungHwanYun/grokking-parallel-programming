// Chapter 7 — Practice on swforces
// Problem 55: Remove Uppercase Letters
// https://swforces.com/problem/55
//
// Reference solution (not printed in the book). Stream compaction on
// characters, the same three steps as Problem 200: flag each lowercase
// letter with 1, run the chapter's Blelloch exclusive scan over the flags
// (padded to the next power of two, as the chapter's Power-of-2 note
// describes), and scatter each keeper to the slot its scan value names.

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void compactLowercase(char* data, char* output, int* count,
                                 int n, int padded_n) {
    __shared__ int buf[256];
    int tid = threadIdx.x;

    // Step 1: Predicate. Flag the characters to keep (lowercase letters)
    int flag = (tid < n && data[tid] >= 'a' && data[tid] <= 'z') ? 1 : 0;
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

    // Step 3: Scatter. Each keeper's destination is its exclusive scan value
    if (tid < n && flag) output[buf[tid]] = data[tid];
    // The last exclusive scan value plus the last flag is the kept count
    if (tid == n - 1) *count = buf[tid] + flag;
}

int main() {
    int n;
    char h_data[128];
    scanf("%d", &n);
    scanf("%s", h_data);

    // Pad to the next power of 2 (from the chapter's Power-of-2 note)
    int padded_n = 1;
    while (padded_n < n) padded_n <<= 1;

    char *d_data, *d_output;
    int* d_count;
    cudaMalloc(&d_data, n * sizeof(char));
    cudaMalloc(&d_output, n * sizeof(char));
    cudaMalloc(&d_count, sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(char), cudaMemcpyHostToDevice);

    compactLowercase<<<1, 256>>>(d_data, d_output, d_count, n, padded_n);

    int m;
    cudaMemcpy(&m, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_data, d_output, m * sizeof(char), cudaMemcpyDeviceToHost);
    h_data[m] = '\0';
    printf("%s\n", h_data);

    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_count);
    return 0;
}
