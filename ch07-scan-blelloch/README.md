# Chapter 7 — Work-Efficient Scan: Blelloch

| File | What it is |
|---|---|
| `01-blelloch-scan.cu` | Complete Blelloch exclusive scan: up-sweep, clear the root, down-sweep; runnable as-is (single block, n must be a power of 2, fixed example data) |
| `02-scan-in-python.py` | The "Scan in PyTorch" listings: `np.cumsum` / `torch.cumsum` and the inclusive-to-exclusive conversion (both listings from that section in one script) |

Practice box at the end of the chapter (reference solutions, not printed in the book). Problem 199 is the chapter's kernel fed from standard input; Problem 200 turns the "Stream compaction" section into working code, with the Power-of-2 note's padding applied to the flag array:

| File | Problem | Link |
|---|---|---|
| `practice/199-exclusive-prefix-sum.cu` | Exclusive Prefix Sum | [swforces.com/problem/199](https://swforces.com/problem/199) |
| `practice/200-stream-compaction.cu` | Stream Compaction | [swforces.com/problem/200](https://swforces.com/problem/200) |
