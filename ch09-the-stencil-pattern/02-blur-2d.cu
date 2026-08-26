// Chapter 9 — The Stencil Pattern
// 2D box blur: 3×3 neighborhood average with the count trick at the edges
// (kernel only; launch with dim3 block(16, 16) as shown in the chapter).

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
