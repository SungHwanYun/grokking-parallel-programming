# Chapter 4 — The Map pattern

| File | Problem | Link |
|---|---|---|
| `01-vector-addition.cu` | Problem 4.1: Vector addition | [swforces.com/problem/4](https://swforces.com/problem/4) |
| `02-scalar-vector-multiplication.cu` | Problem 4.2: Scalar vector multiplication | [swforces.com/problem/39](https://swforces.com/problem/39) |
| `03-vector-relu.cu` | Problem 4.3: Vector ReLU | [swforces.com/problem/195](https://swforces.com/problem/195) |

For Problems 4.2 and 4.3, the book prints the kernel and the launch line and describes the rest of `main()` in prose ("the same five-step pattern as vector addition"); the `main()` functions in those two files were completed to match that description, so every file here is submittable as-is.

Practice box at the end of the chapter (reference solutions, not printed in the book). Every problem is a map in the broad sense: one thread, one output element, no output depending on another. Only Problem 57 is a pure map; in Problem 92 every output past the original length reads the same last character, and in Problem 78 computing each output takes a loop over the whole input:

| File | Problem | Link |
|---|---|---|
| `practice/57-convert-lowercase-to-uppercase.cu` | Convert Lowercase to Uppercase | [swforces.com/problem/57](https://swforces.com/problem/57) |
| `practice/92-string-tail-extension.cu` | String Tail Extension | [swforces.com/problem/92](https://swforces.com/problem/92) |
| `practice/78-multiple-find-element-queries.cu` | Multiple Find Element Queries | [swforces.com/problem/78](https://swforces.com/problem/78) |
