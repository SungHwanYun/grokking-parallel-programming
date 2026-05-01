__global__ void reluKernel(int *A, int *B, int n) {
    int idx = threadIdx.x;
    if(idx < n) {
        B[idx] = (A[idx] <= 0 ? 0 : A[idx]);
    }
}

int main() {
    int n;
    scanf("%d", &n);

    int *h_A, *h_B;
    h_A = (int *)malloc(n * sizeof(int));
    h_B = (int *)malloc(n * sizeof(int));
    for(int i=0; i < n; i++) scanf("%d", &h_A[i]);

    int *d_A, *d_B;
    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));

    cudaMemcpy(d_A, h_A, n * sizeof(int), cudaMemcpyHostToDevice);
    reluKernel<<<1, n>>>(d_A, d_B, n);

    cudaMemcpy(h_B, d_B, n * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0; i < n; i++) printf("%d ", h_B[i]);
    
    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B);
    return 0;
}
