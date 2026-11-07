# Chapter 6 — The Scan pattern: Hillis-Steele

| File | What it is |
|---|---|
| `01-hillis-steele-scan.cu` | The chapter's complete Hillis-Steele inclusive scan, runnable as-is (single block, up to 256 elements, fixed example data) |

Practice box at the end of the chapter (reference solutions, not printed in the book). Problem 198 runs the chapter's kernel against hidden test cases read from standard input. Each of the other three can be solved with one prefix-sum array: the chapter's `scanKernel` computes it, and a short pass over the prefix sums (the lowest point, the balanced split points, the first time step over the threshold) finishes the job:

| File | Problem | Link |
|---|---|---|
| `practice/198-inclusive-prefix-sum.cu` | Inclusive Prefix Sum | [swforces.com/problem/198](https://swforces.com/problem/198) |
| `practice/127-minimum-initial-value.cu` | Minimum Initial Value | [swforces.com/problem/127](https://swforces.com/problem/127) |
| `practice/86-balanced-dual-partition.cu` | Balanced Dual Partition | [swforces.com/problem/86](https://swforces.com/problem/86) |
| `practice/140-prime-time-threshold.cu` | Prime Time Threshold | [swforces.com/problem/140](https://swforces.com/problem/140) |
