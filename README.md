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

- A web browser. That's it: no GPU needed.

The code in this repository is designed to run on [swforces.com](https://swforces.com), the online judge for this book. swforces.com translates your CUDA into CPU-executable code and runs it on regular CPUs, so you can learn and practice parallel programming from any computer.

## Getting Started

Your lab is [swforces.com](https://swforces.com), and setting it up takes only a few minutes:

1. **Create a free account** at [swforces.com](https://swforces.com) (no credit card required).
2. **Open a problem.** Every `.cu` file in this repository links to its problem in the header comment (for example, [swforces.com/problem/191](https://swforces.com/problem/191) for your first Hello, Parallel World).
3. **Test, then Submit.** On a problem page, **Test** runs your code with any input you type (or none) so you can experiment and debug freely; **Submit** grades it against hidden test cases and returns a verdict such as "Accepted" or "Wrong Answer".

Prefer to run code without a problem attached? The **IDE** tab at the top of the site opens a free-standing editor with its own input and output panels.

The full step-by-step walkthrough — creating an account, touring the problem set, submitting your first program, and pointers to free GPU options for running the same code on real hardware — lives online at **[swforces.com/getting-started](https://swforces.com/getting-started)**. Chapter 1's "Your Lab" section introduces the platform as well.

## Optional: Running locally with `nvcc`

If you have an NVIDIA GPU and the CUDA Toolkit installed, you can also compile and run the listings locally:

```
nvcc 01-hello-parallel-world.cu -o hello
./hello
```

The listings use no version-specific features, so any recent CUDA Toolkit works.

## License

MIT. See [LICENSE](LICENSE).
