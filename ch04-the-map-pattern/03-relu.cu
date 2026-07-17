__global__ void reluKernel(float *A, float *B, int n) {
    int idx = threadIdx.x;
    if(idx < n) {
        B[idx] = (A[idx] <= 0 ? 0.0 : A[idx]);
    }
}

int main() {
    int n;
    scanf("%d", &n);

    float *h_A, *h_B;
    h_A = (float *)malloc(n * sizeof(float));
    h_B = (float *)malloc(n * sizeof(float));
    for(int i=0; i < n; i++) scanf("%f", &h_A[i]);

    float *d_A, *d_B;
    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));

    cudaMemcpy(d_A, h_A, n * sizeof(float), cudaMemcpyHostToDevice);
    reluKernel<<<1, n>>>(d_A, d_B, n);

    cudaMemcpy(h_B, d_B, n * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i < n; i++) printf("%.4f ", h_B[i]);
    
    cudaFree(d_A); cudaFree(d_B);
    free(h_A); free(h_B);
    return 0;
}
