# Chapter 10 — Memory access patterns

| File | What it is |
|---|---|
| `01-copy-coalesced.cu` | Coalesced access: consecutive threads read consecutive addresses (kernel only) |
| `02-copy-strided.cu` | Strided access: each thread skips elements (kernel only) |
| `03-copy-scattered.cu` | Scattered access: indices jump around randomly (kernel only) |
| `04-aos-vs-soa.cu` | The `Particle` struct and its SoA counterpart, the `updateX` kernel over two contiguous arrays, and the BEFORE/AFTER pair `updateAoS` / `updateSoA` (kernels only; the `#define N` at the top is not in the book and only makes the listing compile standalone) |
| `05-matrix-sum-rows-cols.cu` | `sumRows` and `sumCols`: row sums look contiguous but stride, column sums look strided but coalesce (kernels only) |

Practice box at the end of the chapter (reference solutions, not printed in the book). In Problem 205, one thread per particle turns the interleaved layout into three contiguous arrays, the transformation behind the chapter's `updateAoS`-to-`updateSoA` rewrite. In Problem 41 the access pattern is the whole point: a plain transpose reads coalesced and writes strided (or the other way round), which is what chapter 11's tiled transpose fixes:

| File | Problem | Link |
|---|---|---|
| `practice/205-aos-to-soa-conversion.cu` | AoS to SoA Conversion | [swforces.com/problem/205](https://swforces.com/problem/205) |
| `practice/41-matrix-transpose.cu` | Matrix Transpose | [swforces.com/problem/41](https://swforces.com/problem/41) |
