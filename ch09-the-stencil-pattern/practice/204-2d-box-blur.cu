// Chapter 9 — Practice on swforces
// Problem 204: 2D Box Blur
// https://swforces.com/problem/204
//
// Reference solution (not printed in the book). The kernel is the chapter's
// blur2D unchanged: a 3x3 window with the count trick, so edges and corners
// average only the pixels that exist. main() launches the 2D grid with
// dim3 block(16, 16) exactly as the chapter shows, one thread per pixel.

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void blur2D(float* input, float* output,
                       int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float sum = 0.0f;
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width &&
                ny >= 0 && ny < height) {
                sum += input[ny * width + nx];
                count++;
            }
        }
    }
    output[y * width + x] = sum / count;
}

int main() {
    int H, W;
    scanf("%d %d", &H, &W);
    float* h_image = (float*)malloc(H * W * sizeof(float));
    for (int i = 0; i < H * W; i++) scanf("%f", &h_image[i]);

    float *d_input, *d_output;
    cudaMalloc(&d_input, H * W * sizeof(float));
    cudaMalloc(&d_output, H * W * sizeof(float));
    cudaMemcpy(d_input, h_image, H * W * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);
    blur2D<<<grid, block>>>(d_input, d_output, W, H);

    cudaMemcpy(h_image, d_output, H * W * sizeof(float), cudaMemcpyDeviceToHost);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) printf("%.6f ", h_image[y * W + x]);
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_image);
    return 0;
}
