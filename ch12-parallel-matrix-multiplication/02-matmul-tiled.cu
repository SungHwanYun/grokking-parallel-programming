// Chapter 12 — Parallel Matrix Multiplication
// Tiled matmul: cooperative tile loading into shared memory with two
// barriers per tile (kernel only; launch with dim3 blockDim(32, 32)).

#define TILE_SIZE 32
__global__ void matmulTiled(float* A, float* B, float* C,
                            int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {

        int aCol = t * TILE_SIZE + threadIdx.x;    // Step 1: Load tiles cooperatively
        tileA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row*K + aCol] : 0.0f;
        int bRow = t * TILE_SIZE + threadIdx.y;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow*N + col] : 0.0f;
        __syncthreads();    // Step 2: Wait for all loads

        for (int k = 0; k < TILE_SIZE; k++)    // Step 3: Compute partial dot product
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();    // Step 4: Wait before next tile
    }
    if (row < M && col < N)
        C[row * N + col] = sum;    // Write final result
}
