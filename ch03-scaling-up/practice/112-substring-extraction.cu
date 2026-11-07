// Chapter 3 — Practice on swforces
// Problem 112: Substring Extraction
// https://swforces.com/problem/112
//
// Reference solution (not printed in the book). The substring has
// len = e - s + 1 characters, so we launch one thread per output character:
// thread idx copies x[s + idx] into output[idx]. The launch uses the
// chapter's recipe (ceiling division for the block count, global index,
// bounds check against len).

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void substringKernel(char* input, char* output, int s, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output[idx] = input[s + idx];
    }
}

int main() {
    int n, s, e;
    char h_input[128], h_output[128];
    scanf("%d", &n);
    scanf("%s", h_input);
    scanf("%d %d", &s, &e);
    int len = e - s + 1;

    char *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(char));
    cudaMalloc(&d_output, len * sizeof(char));
    cudaMemcpy(d_input, h_input, n * sizeof(char), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    substringKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, s, len);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, len * sizeof(char), cudaMemcpyDeviceToHost);
    h_output[len] = '\0';
    printf("%s\n", h_output);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
