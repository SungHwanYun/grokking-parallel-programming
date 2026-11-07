// Chapter 13 — Practice on swforces
// Problem 121: Character Cipher
// https://swforces.com/problem/121
//
// Reference solution (not printed in the book). The three constant-memory
// steps from the chapter applied to a lookup table: declare the five-entry
// table with __constant__ at file scope, copy it in with
// cudaMemcpyToSymbol, and index it in the kernel by each character's offset
// from 'a'. The threads of a warp read at most five distinct addresses,
// all in one cache line, so after the first miss every read is a hit.

#include <stdio.h>
#include <cuda_runtime.h>

// Step 1: Declare in constant memory (file scope)
__constant__ char table[5];

__global__ void cipherKernel(char* input, char* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Step 3: Use it. Each thread looks up its own character
        output[idx] = table[input[idx] - 'a'];
    }
}

int main() {
    int n;
    char h_input[128], h_output[128];
    scanf("%d", &n);
    scanf("%s", h_input);

    char *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(char));
    cudaMalloc(&d_output, n * sizeof(char));
    cudaMemcpy(d_input, h_input, n * sizeof(char), cudaMemcpyHostToDevice);

    // Step 2: Copy the table to constant memory
    char h_table[5] = {'T', 'C', 'P', 'Q', 'A'};    // a, b, c, d, e
    cudaMemcpyToSymbol(table, h_table, 5 * sizeof(char));

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    cipherKernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    cudaMemcpy(h_output, d_output, n * sizeof(char), cudaMemcpyDeviceToHost);
    h_output[n] = '\0';
    printf("%s\n", h_output);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
