# Chapter 12 — Parallel Matrix Multiplication

| File | What it is |
|---|---|
| `01-matmul-naive.cu` | Naive matmul: one thread per output element, every read from global memory (kernel only; launch with `dim3 blockDim(16, 16)` as shown in the chapter) |
| `02-matmul-tiled.cu` | Tiled matmul: cooperative tile loading into shared memory with two barriers per tile (kernel only; launch with `dim3 blockDim(32, 32)`) |
| `03-cublas-sgemm.cu` | The `cublasSgemm` call for row-major C = A × B, exactly as printed in the book (call-site excerpt, not a standalone program) |
