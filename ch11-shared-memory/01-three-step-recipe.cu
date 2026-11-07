// Chapter 11 — Shared memory
// The 3-Step Recipe template: Load, Collaborate, Write back (kernel only).

#define BLOCK_SIZE 256    // change to match your kernel launch configuration
__global__ void recipeTemplate(float* input, float* output) {
    __shared__ float sdata[BLOCK_SIZE];    // __shared__ can't be declared in host code; in this book we always declare it inside the kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Step 1: Load. Everyone copies one piece from Global to Shared
    sdata[threadIdx.x] = input[idx];    // For production code, guard with the identity: sdata[threadIdx.x] = (idx < N) ? input[idx] : 0.0f
    __syncthreads();    // Barrier: all of Step 1 must finish before Step 2 begins
    // Step 2: Collaborate. Reduce in shared memory, one round per stride
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();    // producer-consumer barrier between rounds
    }
    // Step 3: Write back. Thread 0 atomically adds the block's sum to Global
    if (threadIdx.x == 0)
        atomicAdd(output, sdata[0]);    // one atomicAdd per block; initialize *output to zero before launch
}
