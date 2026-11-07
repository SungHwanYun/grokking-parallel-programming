# Chapter 5 — The Reduction pattern

| File | Problem | Link |
|---|---|---|
| `01-reduction-sum.cu` | Problem 5.1: 1D reduction | [swforces.com/problem/49](https://swforces.com/problem/49) |

Practice box at the end of the chapter (reference solutions, not printed in the book). Problem 5.1 fits in a single block, so your multi-block path never really ran; the first two, with n up to 2,000, exercise it. Array Max and Find Minimum also swap the operator, and Count Array Element keeps the sum but changes what each thread contributes:

| File | Problem | Link |
|---|---|---|
| `practice/193-array-sum.cu` | Array Sum | [swforces.com/problem/193](https://swforces.com/problem/193) |
| `practice/194-array-max.cu` | Array Max | [swforces.com/problem/194](https://swforces.com/problem/194) |
| `practice/70-find-minimum.cu` | Find Minimum | [swforces.com/problem/70](https://swforces.com/problem/70) |
| `practice/44-count-array-element.cu` | Count Array Element | [swforces.com/problem/44](https://swforces.com/problem/44) |

In the next four, each thread first turns its element into a value (a match flag, a prime or 0, a fee), and the same tree reduces those values. The last one enumerates candidates in parallel and reduces to the winner with a min reduction:

| File | Problem | Link |
|---|---|---|
| `practice/74-find-element.cu` | Find Element | [swforces.com/problem/74](https://swforces.com/problem/74) |
| `practice/109-sum-of-primes.cu` | Sum of Primes | [swforces.com/problem/109](https://swforces.com/problem/109) |
| `practice/110-phone-bill-calculator.cu` | Phone Bill Calculator | [swforces.com/problem/110](https://swforces.com/problem/110) |
| `practice/115-minimum-sum-for-linear-equation.cu` | Minimum Sum for Linear Equation | [swforces.com/problem/115](https://swforces.com/problem/115) |

`194-array-max.cu` is the complete program built around the book's `maxKernel`, with the `INT_MIN` initialization the chapter explains.
