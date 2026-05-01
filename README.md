# Grokking Parallel Programming: With Examples in CUDA

Code listings for [*Grokking Parallel Programming*](https://www.manning.com/books/grokking-parallel-programming) by SungHwan Yun (Manning Publications).

## Repository Structure

Each chapter has its own folder. Files within a chapter are numbered in the order they appear in the book.

- `ch01-why-parallel-programming/`
- `ch02-first-parallel-program/`
- ... (17 chapters + 2 appendices)

## Requirements

- A web browser. That's it — no GPU needed!

The code in this repository is designed to run on [cudaforces.com](https://cudaforces.com), the online judge for this book. cudaforces.com transpiles your CUDA code to C++ and runs it on regular CPUs, so you can learn and practice parallel programming from any computer.

## Running the Code

The recommended way to run the listings is to submit them to cudaforces.com:

1. Visit [cudaforces.com](https://cudaforces.com) and create a free account (no credit card or email verification required).
2. Open the problem corresponding to the listing.
3. Paste the code, submit, and get instant feedback.

### Optional: Running locally with `nvcc`

If you have an NVIDIA GPU and the CUDA Toolkit installed, you can also compile and run the listings locally:

```bash
nvcc 01-hello-kernel.cu -o hello
./hello
```

This requires CUDA Toolkit 12.0 or later and an NVIDIA GPU with compute capability 6.0 or later.

## License

MIT — see [LICENSE](LICENSE).
