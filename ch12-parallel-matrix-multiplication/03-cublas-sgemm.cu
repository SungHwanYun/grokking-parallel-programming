// Chapter 12 — Parallel Matrix Multiplication
// In production: the cublasSgemm call for row-major C = A × B.
// Call-site excerpt as printed in the book, not a standalone program.

#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

float alpha = 1.0f, beta = 0.0f;

cublasSgemm(handle,
    // transpose flags: CUBLAS_OP_N means “no transpose” (use the matrix as-is)
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,    // dimensions (note: N first!)
    &alpha,
    d_B, N,    // B matrix, leading dimension
    d_A, K,    // A matrix, leading dimension
    &beta,
    d_C, N);    // C matrix, leading dimension

cublasDestroy(handle);
