# Chapter 8 — The Histogram Pattern

| File | What it is |
|---|---|
| `01-count-positive.cu` | Complete program: count positive numbers with one atomic counter, runnable as-is (fixed example data; prints `Positive count: 5`) |
| `02-histogram-atomic.cu` | A first fix: 256-bin histogram with global atomics (kernel only) |
| `03-histogram-shared.cu` | A faster fix: privatization, a per-block histogram in shared memory merged to global once per bin (kernel only) |

Practice box at the end of the chapter (reference solutions, not printed in the book). Both use the chapter's kernels unchanged; n goes up to 2,048, so the launches genuinely run multiple blocks:

| File | Problem | Link |
|---|---|---|
| `practice/201-count-positive-numbers.cu` | Count Positive Numbers | [swforces.com/problem/201](https://swforces.com/problem/201) |
| `practice/202-histogram.cu` | Histogram | [swforces.com/problem/202](https://swforces.com/problem/202) |
