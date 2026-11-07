# Chapter 12 — Parallel matrix multiplication

| File | What it is |
|---|---|
| `01-matmul-naive.cu` | Naive matmul: one thread per output element, every read from global memory (kernel only; launch with `dim3 blockDim(16, 16)` as shown in the chapter) |
| `02-matmul-tiled.cu` | Tiled matmul: cooperative tile loading into shared memory with two barriers per tile (kernel only; launch with `dim3 blockDim(TILE_SIZE, TILE_SIZE)`) |
| `03-cublas-sgemm.cu` | The `cublasSgemm` call for row-major C = A × B, exactly as printed in the book (call-site excerpt, not a standalone program) |

Practice box at the end of the chapter (reference solutions, not printed in the book). Problem 25 is the pattern itself: the naive kernel, one thread per output element, on a matrix small enough to check by hand. Problem 111 is the twist: replace multiply with add and add with min, and the same triple loop extends every path by one more road; repeat it and you have shortest paths:

| File | Problem | Link |
|---|---|---|
| `practice/25-matrix-multiplication.cu` | Matrix Multiplication | [swforces.com/problem/25](https://swforces.com/problem/25) |
| `practice/111-shortest-path.cu` | Shortest Path | [swforces.com/problem/111](https://swforces.com/problem/111) |
