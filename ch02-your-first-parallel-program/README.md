# Chapter 2 — Your first parallel program

Complete programs from the chapter, in the order they appear:

| File | Problem | Link |
|---|---|---|
| `01-hello-parallel-world.cu` | Problem 2.1: Print Hello, Parallel World! 8 times | [swforces.com/problem/191](https://swforces.com/problem/191) |
| `02-thread-identity.cu` | Problem 2.2: Thread identity | [swforces.com/problem/189](https://swforces.com/problem/189) |
| `03-square-array-elements.cu` | Problem 2.3: Square array elements | [swforces.com/problem/190](https://swforces.com/problem/190) |

Practice box at the end of the chapter (reference solutions, not printed in the book). All of these fit in a single block; most add a small twist to the same five-step pattern:

| File | Problem | Link |
|---|---|---|
| `practice/197-double-array-elements.cu` | Double Array Elements | [swforces.com/problem/197](https://swforces.com/problem/197) |
| `practice/3-print-hello-world-cuda-n-times.cu` | Print Hello World Cuda N Times | [swforces.com/problem/3](https://swforces.com/problem/3) |
| `practice/18-echo-string.cu` | Echo String | [swforces.com/problem/18](https://swforces.com/problem/18) |
| `practice/21-sum-of-two-integers.cu` | Sum of Two Integers | [swforces.com/problem/21](https://swforces.com/problem/21) |
| `practice/33-greater-than.cu` | Greater Than | [swforces.com/problem/33](https://swforces.com/problem/33) |
| `practice/7-swap-two-variables.cu` | Swap Two Variables | [swforces.com/problem/7](https://swforces.com/problem/7) |
| `practice/12-sort-three-numbers.cu` | Sort Three Numbers | [swforces.com/problem/12](https://swforces.com/problem/12) |
| `practice/103-letter-grade.cu` | Letter Grade | [swforces.com/problem/103](https://swforces.com/problem/103) |
| `practice/26-caesar-cipher-decryption.cu` | Caesar Cipher Decryption | [swforces.com/problem/26](https://swforces.com/problem/26) |
| `practice/75-coding-time-to-minutes.cu` | Coding Time to Minutes | [swforces.com/problem/75](https://swforces.com/problem/75) |

The practice solutions stay within this chapter's toolkit: one block, `threadIdx.x`, scalar kernel arguments, and the allocate / copy in / compute / copy out / free pattern. Where a problem has a single answer (a sum, a letter, a character), one thread computes it and the host prints it; where the output is a string, one thread handles one character and the host prints the copied-back array, because a real GPU gives no guarantee about the order in which threads reach `printf`.

Shorter in-text snippets (the kernel on its own, the RIGHT versions from the Common mistakes pairs) all appear inside these complete programs.

Note: the book omits CUDA error checking in this chapter to keep examples short; proper error handling is introduced in chapter 4.
