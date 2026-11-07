# Chapter 3 — Scaling up: from one block to many

| File | Problem | Link |
|---|---|---|
| `01-array-double.cu` | Problem 3.1: Array double | [swforces.com/problem/192](https://swforces.com/problem/192) |

The complete program demonstrates the chapter's four-step thread-to-element recipe: pick a block size, compute the block count with ceiling division, compute the global index `blockIdx.x * blockDim.x + threadIdx.x`, and guard with the bounds check `if (idx < n)`.

Practice box at the end of the chapter (reference solutions, not printed in the book). The inputs are small enough for one block, but, as the chapter asks, every solution uses a multi-block launch with the global index, ceiling division for the block count, and the bounds check:

| File | Problem | Link |
|---|---|---|
| `practice/6-class-grade-report.cu` | Class Grade Report | [swforces.com/problem/6](https://swforces.com/problem/6) |
| `practice/112-substring-extraction.cu` | Substring Extraction | [swforces.com/problem/112](https://swforces.com/problem/112) |
| `practice/43-array-reverse.cu` | Array Reverse | [swforces.com/problem/43](https://swforces.com/problem/43) |
| `practice/53-extract-even-position-characters.cu` | Extract Even-Position Characters | [swforces.com/problem/53](https://swforces.com/problem/53) |
