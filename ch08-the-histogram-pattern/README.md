# Chapter 8 — The Histogram pattern

| File | What it is |
|---|---|
| `01-count-positive.cu` | Complete program: count positive numbers with one atomic counter, runnable as-is (fixed example data; prints `Positive count: 5`) |
| `02-histogram-atomic.cu` | A first fix: 256-bin histogram with global atomics (kernel only) |
| `03-histogram-shared.cu` | A faster fix: privatization, a per-block histogram in shared memory merged to global once per bin (kernel only) |

Practice box at the end of the chapter (reference solutions, not printed in the book). Problems 201 and 202 are the two kernels from this chapter, run against standard input with n up to 2,048 so the launches use several blocks. The next three count other things: survey answers (a three-bin histogram), the letters of a string (a counting sort built on `histogramShared`), and the meetings running at the same time (a histogram over time slots, then a max reduction). Problem 207 is a max reduction on floats whose merge needs the `atomicMaxFloat` from the chapter's sidebar:

| File | Problem | Link |
|---|---|---|
| `practice/201-count-positive-numbers.cu` | Count Positive Numbers | [swforces.com/problem/201](https://swforces.com/problem/201) |
| `practice/202-histogram.cu` | Histogram | [swforces.com/problem/202](https://swforces.com/problem/202) |
| `practice/87-favorite-subject-survey.cu` | Favorite Subject Survey | [swforces.com/problem/87](https://swforces.com/problem/87) |
| `practice/95-sort-string.cu` | Sort String | [swforces.com/problem/95](https://swforces.com/problem/95) |
| `practice/155-minimum-meeting-rooms.cu` | Minimum Meeting Rooms | [swforces.com/problem/155](https://swforces.com/problem/155) |
| `practice/207-float-max-reduction.cu` | Float Max Reduction | [swforces.com/problem/207](https://swforces.com/problem/207) |
