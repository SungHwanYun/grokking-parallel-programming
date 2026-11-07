// Chapter 2 — Practice on swforces
// Problem 103: Letter Grade
// https://swforces.com/problem/103
//
// Reference solution (not printed in the book). One thread turns the score
// into a letter with a chain of if/else, the same kind of branch as the
// chapter's if (idx < n), and writes the character to device memory for
// the host to print.

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void gradeKernel(int score, char* grade) {
    if (score >= 90)      *grade = 'A';
    else if (score >= 80) *grade = 'B';
    else if (score >= 70) *grade = 'C';
    else if (score >= 60) *grade = 'D';
    else                  *grade = 'E';
}

int main() {
    int score;
    scanf("%d", &score);

    char* d_grade;
    cudaMalloc(&d_grade, sizeof(char));

    gradeKernel<<<1, 1>>>(score, d_grade);
    cudaDeviceSynchronize();

    char h_grade;
    cudaMemcpy(&h_grade, d_grade, sizeof(char), cudaMemcpyDeviceToHost);
    printf("%c\n", h_grade);

    cudaFree(d_grade);
    return 0;
}
