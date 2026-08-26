// Chapter 8 — The Histogram Pattern
// A faster fix: privatization — per-block histogram in shared memory,
// then one atomic merge per bin per block (kernel only).

__global__ void histogramShared(int* data, int* hist, int n) {
    __shared__ int localHist[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Initialize
    for (int i = tid; i < 256; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();

    // Step 2: Build local histogram
    if (idx < n) {
        int bin = data[idx];
        atomicAdd(&localHist[bin], 1);
    }
    __syncthreads();

    // Step 3: Merge to global
    for (int i = tid; i < 256; i += blockDim.x) {
        if (localHist[i] > 0) atomicAdd(&hist[i], localHist[i]);
    }
}
