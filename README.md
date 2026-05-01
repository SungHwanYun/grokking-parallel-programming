# Grokking Parallel Programming: With Examples in CUDA

Code listings for [*Grokking Parallel Programming*](https://www.manning.com/books/grokking-parallel-programming) by SungHwan Yun (Manning Publications).

## Repository Structure

Each chapter has its own folder. Files within a chapter are numbered in the order they appear in the book.

- `ch01-why-parallel-programming/`
- `ch02-first-parallel-program/`
- ... (17 chapters + 2 appendices)

## Requirements

- CUDA Toolkit 12.0 or later
- NVIDIA GPU with compute capability 6.0 or later

## Building

Each `.cu` file can be compiled with `nvcc`:

\```bash
nvcc 01-hello-kernel.cu -o hello
./hello
\```

## Companion Online Judge

Readers can submit and run their solutions at [cudaforces.com](https://cudaforces.com), the online judge maintained for this book.

## License

MIT — see [LICENSE](LICENSE).
