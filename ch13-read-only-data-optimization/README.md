# Chapter 13 — Read-only data optimization

| File | Problem | Link |
|---|---|---|
| `01-scale-array-constant.cu` | Problem 13.1: Scale array with constant factor, as printed in the chapter (example values hard-coded) | [swforces.com/problem/196](https://swforces.com/problem/196) |
| `02-scale-array-constant-stdin.cu` | The same program changed as the chapter describes for submission: input read from stdin, results printed with `%g` | [swforces.com/problem/196](https://swforces.com/problem/196) |

The program follows the chapter's three steps for constant memory: declare with `__constant__` at file scope, copy from the host with `cudaMemcpyToSymbol`, and read it in the kernel, where the constant cache broadcasts one fetch to the whole warp.

Practice box at the end of the chapter (reference solution, not printed in the book). A five-entry lookup table that fits in one cache line still works well in constant memory, even though each thread indexes it by its own character:

| File | Problem | Link |
|---|---|---|
| `practice/121-character-cipher.cu` | Character Cipher | [swforces.com/problem/121](https://swforces.com/problem/121) |
