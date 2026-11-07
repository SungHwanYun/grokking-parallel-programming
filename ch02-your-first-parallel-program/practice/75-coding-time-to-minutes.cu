// Chapter 2 — Practice on swforces
// Problem 75: Coding Time to Minutes
// https://swforces.com/problem/75
//
// Reference solution (not printed in the book). The five characters of
// "HH:MM" are copied to the device as a char array, and one thread does the
// digit arithmetic: each digit is its character minus '0', and the total is
// 60 * HH + MM.

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void minutesKernel(char* time, int* result) {
    int hours   = (time[0] - '0') * 10 + (time[1] - '0');    // "HH"
    int minutes = (time[3] - '0') * 10 + (time[4] - '0');    // "MM" (time[2] is the colon)
    *result = hours * 60 + minutes;
}

int main() {
    char h_time[16];
    scanf("%s", h_time);

    char* d_time;
    int* d_result;
    cudaMalloc(&d_time, 5 * sizeof(char));
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_time, h_time, 5 * sizeof(char), cudaMemcpyHostToDevice);

    minutesKernel<<<1, 1>>>(d_time, d_result);
    cudaDeviceSynchronize();

    int h_result;
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d\n", h_result);

    cudaFree(d_time);
    cudaFree(d_result);
    return 0;
}
