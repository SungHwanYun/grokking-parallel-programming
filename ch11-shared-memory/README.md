# Chapter 11 — Shared Memory

| File | What it is |
|---|---|
| `01-three-step-recipe.cu` | The 3-Step Recipe template: Load, Collaborate, Write back (kernel only) |
| `02-transpose-tiled.cu` | A first taste of tiling: matrix transpose through a padded 32×33 tile, coalesced on both the read and the write (kernel only) |

Exercise 11.3's `blockSum` kernel is deliberately printed without its barriers, so it lives in the book (and its solution in Appendix A), not here.

Practice box at the end of the chapter (reference solution, not printed in the book). The 3-Step Recipe applied to a new problem: Load each thread's product into shared memory, Collaborate on Chapter 5's tree reduction, Write back one partial sum per block with `atomicAdd`:

| File | Problem | Link |
|---|---|---|
| `practice/206-dot-product.cu` | Dot Product | [swforces.com/problem/206](https://swforces.com/problem/206) |
