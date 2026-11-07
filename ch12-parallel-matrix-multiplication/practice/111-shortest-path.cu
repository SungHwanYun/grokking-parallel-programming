// Chapter 12 — Practice on swforces
// Problem 111: Shortest Path
// https://swforces.com/problem/111
//
// Reference solution (not printed in the book). Matrix multiplication with
// the operators swapped: multiply becomes add and add becomes min. If
// D[i][j] is the shortest distance using paths of at most r roads and W is
// the road-length matrix (INF where there is no road), then
//     D'[i][j] = min(D[i][j], min over k of D[i][k] + W[k][j])
// is the shortest distance using at most r + 1 roads: the same triple loop
// as matmulNaive, one thread per output element, extending every path by
// one more road. A shortest path uses at most N - 1 roads, so the kernel
// runs N - 1 times, reading one buffer and writing the other.

#include <stdio.h>
#include <cuda_runtime.h>

#define INF 1000000000    // Larger than any real distance (at most 50 roads of 10,000)
#define MAX_N 16

__global__ void minPlusKernel(int* D, int* W, int* Dnext, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int best = D[row * N + col];                  // Keep the current distance as a candidate
        for (int k = 0; k < N; k++) {
            if (D[row * N + k] < INF && W[k * N + col] < INF) {    // Only real paths and real roads
                int candidate = D[row * N + k] + W[k * N + col];   // "multiply" is add
                if (candidate < best) best = candidate;            // "add" is min
            }
        }
        Dnext[row * N + col] = best;
    }
}

int main() {
    int N, M;
    scanf("%d %d", &N, &M);
    int h_W[MAX_N * MAX_N];
    for (int i = 0; i < N * N; i++) h_W[i] = INF;
    for (int i = 0; i < N; i++) h_W[i * N + i] = 0;    // Staying put costs nothing
    for (int e = 0; e < M; e++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        h_W[(u - 1) * N + (v - 1)] = w;                  // One-way road u -> v
    }
    int X, Z;
    scanf("%d %d", &X, &Z);

    int *d_W, *d_D, *d_Dnext;
    cudaMalloc(&d_W, N * N * sizeof(int));
    cudaMalloc(&d_D, N * N * sizeof(int));
    cudaMalloc(&d_Dnext, N * N * sizeof(int));
    cudaMemcpy(d_W, h_W, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_W, N * N * sizeof(int), cudaMemcpyHostToDevice);    // Paths of at most one road

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);
    for (int r = 1; r < N; r++) {                        // Each round allows one more road
        minPlusKernel<<<gridDim, blockDim>>>(d_D, d_W, d_Dnext, N);
        int* tmp = d_D; d_D = d_Dnext; d_Dnext = tmp;    // The new distances become the input
    }

    int h_D[MAX_N * MAX_N];
    cudaMemcpy(h_D, d_D, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    int answer = h_D[(X - 1) * N + (Z - 1)];
    printf("%d\n", answer < INF ? answer : -1);

    cudaFree(d_W);
    cudaFree(d_D);
    cudaFree(d_Dnext);
    return 0;
}
