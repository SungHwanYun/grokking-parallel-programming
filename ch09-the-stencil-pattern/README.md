# Chapter 9 — The Stencil Pattern

| File | What it is |
|---|---|
| `01-average-1d.cu` | 1D stencil: 3-point moving average with boundary handling (kernel only) |
| `02-blur-2d.cu` | 2D box blur: 3×3 neighborhood average with the count trick at the edges (kernel only; launch with `dim3 block(16, 16)` as shown in the chapter) |

The chapter presents these as kernels; the boundary-strategy helpers (clamp, zero padding, skip, wrap-around) appear in the book as short fragments and are not duplicated here.

Practice box at the end of the chapter (reference solutions, not printed in the book). Both wrap the chapter's kernels in a `main()` that reads from standard input; the 2D blur launches the `dim3 block(16, 16)` grid from the chapter:

| File | Problem | Link |
|---|---|---|
| `practice/203-1d-moving-average.cu` | 1D Moving Average | [swforces.com/problem/203](https://swforces.com/problem/203) |
| `practice/204-2d-box-blur.cu` | 2D Box Blur | [swforces.com/problem/204](https://swforces.com/problem/204) |
