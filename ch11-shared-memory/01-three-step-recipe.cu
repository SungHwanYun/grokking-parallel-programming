// Chapter 11 — Shared Memory
// The 3-Step Recipe template: Load, Collaborate, Write back (kernel only).

#define BLOCK_SIZE 256    // change to match your kernel launch configuration
__global__ void recipeTemplate(float* input, float* output) {
    // __shared__ can only appear inside a __global__ or __device__ function, never at file scope
    __shared__ float sdata[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Step 1: Load — everyone copies one piece from Global to Shared
    // For production code, add: if (idx < N) to guard the array boundary
    sdata[threadIdx.x] = input[idx];
    __syncthreads();    // CRITICAL barrier: all of Step 1 must finish before Step 2 begins
    // Step 2: Collaborate — reduce in shared memory, one round per stride
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();    // producer-consumer barrier between rounds
    }
    // Step 3: Write back — thread 0 atomically adds the block’s sum to Global
    if (threadIdx.x == 0)
        // one atomicAdd per block; initialize *output to zero before launch
        atomicAdd(output, sdata[0]);
}
