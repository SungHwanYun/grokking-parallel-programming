// Chapter 2 — Practice on swforces
// Problem 26: Caesar Cipher Decryption
// https://swforces.com/problem/26
//
// Reference solution (not printed in the book). Character arithmetic in a
// one-thread kernel: the letter's position in the alphabet (a - 'a') is
// shifted by k modulo 26, then rebased on 'A' to produce the uppercase
// result. k can be as large as 10^8, so the shift is reduced with % 26
// rather than by looping.

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void cipherKernel(char letter, int k, char* result) {
    int position = letter - 'a';                  // 0 for 'a', ..., 25 for 'z'
    *result = 'A' + (position + k % 26) % 26;     // Shift, wrap around, make uppercase
}

int main() {
    char letter;
    int k;
    scanf(" %c %d", &letter, &k);

    char* d_result;
    cudaMalloc(&d_result, sizeof(char));

    cipherKernel<<<1, 1>>>(letter, k, d_result);
    cudaDeviceSynchronize();

    char h_result;
    cudaMemcpy(&h_result, d_result, sizeof(char), cudaMemcpyDeviceToHost);
    printf("%c\n", h_result);

    cudaFree(d_result);
    return 0;
}
