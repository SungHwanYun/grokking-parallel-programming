# Chapter 3 — Scaling up: from one block to many

| File | Problem | Link |
|---|---|---|
| `01-array-double.cu` | Problem 3.1: Array Double | [swforces.com/problem/192](https://swforces.com/problem/192) |

The complete program demonstrates the four-step multi-block recipe: pick a block size, compute the block count with ceiling division, compute the global index `blockIdx.x * blockDim.x + threadIdx.x`, and guard with the bounds check `if (idx < n)`.
