// Chapter 12 — Parallel Matrix Multiplication
// Naive matmul: one thread per output element, all reads from global memory
// (kernel only; launch with dim3 blockDim(16, 16) as shown in the chapter).

__global__ void matmulNaive(float* A, float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
