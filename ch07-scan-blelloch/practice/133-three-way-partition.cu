// Chapter 7 — Practice on swforces
// Problem 133: Three-Way Partition
// https://swforces.com/problem/133
//
// Reference solution (not printed in the book). Three compactions into one
// output array. Each thread flags its element as less than, equal to, or
// greater than k; three exclusive scans (one per group, each the chapter's
// Blelloch scan over a padded flag array) tell every element how many
// group-mates precede it. The groups are then laid out back to back:
// less-than elements start at 0, equal elements after the less-than count,
// greater-than elements after both. Scanning preserves the original order
// inside each group, which is what the problem asks for.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Blelloch exclusive scan of buf[0 .. padded_n - 1] in shared memory
__device__ void exclusiveScan(int* buf, int tid, int padded_n) {
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
}

__global__ void partitionKernel(int* data, int* output, int k, int n, int padded_n) {
    __shared__ int less[256];
    __shared__ int equal[256];
    __shared__ int greater[256];
    int tid = threadIdx.x;

    // Step 1: Predicates. One flag array per group
    int value = (tid < n) ? data[tid] : 0;
    int isLess    = (tid < n && value <  k) ? 1 : 0;
    int isEqual   = (tid < n && value == k) ? 1 : 0;
    int isGreater = (tid < n && value >  k) ? 1 : 0;
    less[tid]    = (tid < padded_n) ? isLess    : 0;
    equal[tid]   = (tid < padded_n) ? isEqual   : 0;
    greater[tid] = (tid < padded_n) ? isGreater : 0;
    __syncthreads();

    // Step 2: Three exclusive scans
    exclusiveScan(less, tid, padded_n);
    exclusiveScan(equal, tid, padded_n);
    exclusiveScan(greater, tid, padded_n);

    // Group sizes: the last scan value plus the last flag
    int lastValue = data[n - 1];
    int numLess  = less[n - 1]  + (lastValue <  k ? 1 : 0);
    int numEqual = equal[n - 1] + (lastValue == k ? 1 : 0);

    // Step 3: Scatter. Each group starts where the previous one ends
    if (tid < n) {
        if (isLess)    output[less[tid]] = value;
        if (isEqual)   output[numLess + equal[tid]] = value;
        if (isGreater) output[numLess + numEqual + greater[tid]] = value;
    }
}

int main() {
    int n, k;
    scanf("%d %d", &n, &k);
    int* h_data = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) scanf("%d", &h_data[i]);

    // Pad to the next power of 2 (from the chapter's Power-of-2 note)
    int padded_n = 1;
    while (padded_n < n) padded_n <<= 1;

    int *d_data, *d_output;
    cudaMalloc(&d_data, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    partitionKernel<<<1, 256>>>(d_data, d_output, k, n, padded_n);

    cudaMemcpy(h_data, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) printf("%d ", h_data[i]);
    printf("\n");

    cudaFree(d_data);
    cudaFree(d_output);
    free(h_data);
    return 0;
}
