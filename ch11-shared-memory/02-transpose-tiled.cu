// Chapter 11 — Shared Memory
// A first taste of tiling: matrix transpose with a padded 32×33 tile,
// coalesced on both the read and the write (kernel only).

__global__ void transposeTiled(const float* in, float* out, int n) {
    __shared__ float tile[32][33];    // Padding: 33 columns, not 32
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    tile[threadIdx.y][threadIdx.x] = in[y * n + x];    // Coalesced read from global memory
    __syncthreads();    // Barrier between load and write back
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    // Coalesced write of the transposed tile
    out[y * n + x] = tile[threadIdx.x][threadIdx.y];
}
