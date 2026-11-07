# Chapter 11 — Shared memory

| File | What it is |
|---|---|
| `01-three-step-recipe.cu` | The 3-Step Recipe template: Load, Collaborate, Write back (kernel only) |
| `02-transpose-tiled.cu` | A first taste of tiling: matrix transpose through a padded 32×33 tile, coalesced on both the read and the write (kernel only) |

Exercise 11.3's `blockSum` kernel is deliberately printed without its barriers, so it lives in the book (and its solution in Appendix A), not here.

Practice box at the end of the chapter (reference solutions, not printed in the book). Problem 206 is the 3-Step Recipe applied to a new problem: Load each thread's product into shared memory, Collaborate on chapter 5's tree reduction, Write back one partial sum per block with `atomicAdd`. In Problem 67 overlapping windows mean data reuse, and in Problem 164 a student table loaded once serves every query in the block; both are jobs for shared memory:

| File | Problem | Link |
|---|---|---|
| `practice/206-dot-product.cu` | Dot Product | [swforces.com/problem/206](https://swforces.com/problem/206) |
| `practice/67-max-subarray-sum.cu` | Max Subarray Sum | [swforces.com/problem/67](https://swforces.com/problem/67) |
| `practice/164-student-preference-query.cu` | Student Preference Query | [swforces.com/problem/164](https://swforces.com/problem/164) |
