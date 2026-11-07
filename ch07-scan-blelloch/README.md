# Chapter 7 — Work-efficient scan: Blelloch

| File | What it is |
|---|---|
| `01-blelloch-scan.cu` | The chapter's complete Blelloch exclusive scan: up-sweep, clear the root, down-sweep; runnable as-is (single block, n must be a power of 2, fixed example data) |
| `02-scan-in-python.py` | The "Scan in PyTorch" listings: `np.cumsum` / `torch.cumsum` and the inclusive-to-exclusive conversion (both listings from that section in one script) |

Practice box at the end of the chapter (reference solutions, not printed in the book). Problem 199 is the chapter's kernel fed from standard input; Problem 200 turns the "Stream compaction" section into working code, with the Power-of-2 note's padding applied to the flag array. Problems 55 and 133 are more compactions with the same scan machinery underneath: 55 keeps the lowercase letters, and 133 runs three compactions (less than, equal to, greater than k) into one output array, laying the groups out back to back:

| File | Problem | Link |
|---|---|---|
| `practice/199-exclusive-prefix-sum.cu` | Exclusive Prefix Sum | [swforces.com/problem/199](https://swforces.com/problem/199) |
| `practice/200-stream-compaction.cu` | Stream Compaction | [swforces.com/problem/200](https://swforces.com/problem/200) |
| `practice/55-remove-uppercase-letters.cu` | Remove Uppercase Letters | [swforces.com/problem/55](https://swforces.com/problem/55) |
| `practice/133-three-way-partition.cu` | Three-Way Partition | [swforces.com/problem/133](https://swforces.com/problem/133) |
