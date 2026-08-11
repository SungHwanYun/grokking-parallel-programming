# Chapter 5 — The Reduction pattern

| File | Problem | Link |
|---|---|---|
| `01-reduction-sum.cu` | Problem 5.1: 1D Reduction | [swforces.com/problem/49](https://swforces.com/problem/49) |

Practice box at the end of the chapter (reference solutions, not printed in the book). Problem 5.1 fits in a single block, so your multi-block path never really ran; these four exercise it, and the last three swap the operator:

| File | Problem | Link |
|---|---|---|
| `practice/193-array-sum.cu` | Array Sum | [swforces.com/problem/193](https://swforces.com/problem/193) |
| `practice/194-array-max.cu` | Array Max | [swforces.com/problem/194](https://swforces.com/problem/194) |
| `practice/70-find-minimum.cu` | Find Minimum | [swforces.com/problem/70](https://swforces.com/problem/70) |
| `practice/44-count-array-element.cu` | Count Array Element | [swforces.com/problem/44](https://swforces.com/problem/44) |

`194-array-max.cu` is the complete program built around the book's `maxKernel` (with the `INT_MIN` initialization the chapter explains); the min and count variants follow the same swap-the-operator recipe.
