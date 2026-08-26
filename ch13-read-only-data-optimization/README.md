# Chapter 13 — Read-Only Data Optimization

| File | Problem | Link |
|---|---|---|
| `01-scale-array-constant.cu` | Problem 13.1: Scale Array with Constant Factor | [swforces.com/problem/196](https://swforces.com/problem/196) |

The complete program follows the chapter's 3 steps for constant memory: declare with `__constant__` at file scope, copy from the host with `cudaMemcpyToSymbol`, and read it in the kernel where the constant cache broadcasts one fetch to the whole warp. The `#include <stdio.h>` at the top is not printed in the book's listing; it is required for `printf` on the host side.
