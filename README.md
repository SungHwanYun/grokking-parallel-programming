# Grokking Parallel Programming: With Examples in CUDA

Code listings for [*Grokking Parallel Programming*](https://www.manning.com/books/grokking-parallel-programming) by SungHwan Yun (Manning Publications).

## Repository Structure

Each chapter has its own folder. Files within a chapter are numbered in the order they appear in the book, and each file's header comment links to the swforces problem it solves. Where a chapter ends with a **Practice on swforces** box, reference solutions for those extra problems live in that chapter's `practice/` subfolder.

The book has three parts and 14 chapters:

- **Part 1 — Getting Started** (Chapters 1–3)
- **Part 2 — Fundamental Parallel Patterns** (Chapters 4–9): Map, Reduce, Scan, Histogram, Stencil
- **Part 3 — Mastering GPU Performance** (Chapters 10–14)

Code is currently available for Chapters 1–5, matching the chapters released in the MEAP (Manning Early Access Program). Folders for the remaining chapters will be added as they are released.

- `ch01-why-parallel-programming/` — conceptual chapter, no runnable code (see its README)
- `ch02-your-first-parallel-program/`
- `ch03-scaling-up/`
- `ch04-the-map-pattern/`
- `ch05-the-reduction-pattern/`

## Requirements

- A web browser. That's it — no GPU needed!

The code in this repository is designed to run on [swforces.com](https://swforces.com), the online judge for this book. swforces.com translates your CUDA code to C++ and runs it on regular CPUs, so you can learn and practice parallel programming from any computer.

## Running the Code

Visit [swforces.com](https://swforces.com), create a free account, and open the problem linked at the top of each `.cu` file. Two buttons let you run your code:

- **Test** — your debugging playground. Paste the code, type any input you'd like (or leave it empty), and click Test to see the output immediately. Use this to experiment and debug freely.
- **Submit** — grades your code against hidden test cases and returns a verdict ("Accepted", "Wrong Answer", etc.). Use this when you're confident your solution is correct.

You can also run any code without a problem attached: the **IDE** tab at the top of the site opens a free-standing editor with its own input and output panels.

New to the platform? The step-by-step walkthrough lives at [swforces.com/getting-started](https://swforces.com/getting-started).

### Optional: Running locally with `nvcc`

If you have an NVIDIA GPU and the CUDA Toolkit installed, you can also compile and run the listings locally:

```
nvcc 01-hello-parallel-world.cu -o hello
./hello
```

This requires CUDA Toolkit 12.0 or later and an NVIDIA GPU with compute capability 6.0 or later.

## License

MIT — see [LICENSE](LICENSE).
