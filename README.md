# Grokking Parallel Programming: With Examples in CUDA

Code listings for [*Grokking Parallel Programming*](https://www.manning.com/books/grokking-parallel-programming)
by SungHwan Yun (Manning Publications).

## Repository Structure

```
.
├── common/                  # Shared infrastructure (header + template + guide)
│   ├── common.h             # Error checking, timing, benchmarking helpers
│   ├── template.cu          # Reference benchmark template
│   └── README.md            # Coding & measurement guide
├── colab/                   # Per-chapter Colab measurement notebooks
│   ├── measurement_template.ipynb
│   ├── ch04_map.ipynb
│   └── ...
├── ch01-why-parallel-programming/
├── ch02-your-first-parallel-program/
├── ...
├── ch17-from-practice-to-production/
├── appendix-a-cuda-reference/
├── appendix-b-exercise-answers/
└── README.md
```

Each chapter folder contains the code listings in the order they appear in
the book. The `common/` folder holds the infrastructure those listings
build on. The `colab/` folder holds notebooks that measure and report
kernel performance.

## How to run the code

You have two options, and we recommend doing **both** in order. Each
catches different problems.

### Stage 1 — Correctness on cudaforces.com

Best for confirming your algorithm is right, no setup required.

[cudaforces.com](https://cudaforces.com) is the online judge built for this
book. Create a free account (no email or card required), open the problem
matching the listing, and click:

- **Test** — try your code on input you type yourself.
- **Submit** — grade against hidden test cases.

cudaforces.com transpiles CUDA to C++ and runs it on regular CPUs, so it
checks behavior rather than performance. Almost all listings in chapters
1–13 can be verified here.

### Stage 2 — Performance on Google Colab (free T4)

Best for seeing actual GPU performance and reproducing the book's numbers.

Open the matching notebook from `colab/` in [Google Colab](https://colab.research.google.com),
verify you got a T4, and run all cells. The notebook will:

1. Confirm the GPU is a T4.
2. Pull `common/common.h` and the chapter's `.cu` file from this repo.
3. Compile with `nvcc -O3 -std=c++17 -arch=sm_75`.
4. Sweep over N and block size, recording kernel time, bandwidth, and
   GFLOPS for each combination.
5. Print a results table that should roughly match the book's table for
   the same chapter.

The book's reference numbers are measured on the **same free Colab T4**
hardware you will use, so your numbers should be within a few percent of
the book's. If they are not, the most likely cause is that Colab gave you
a non-T4 GPU — disconnect and reconnect.

### Stage 3 (optional) — Locally with your own NVIDIA GPU

If you have an NVIDIA GPU and CUDA Toolkit 12.0 or later installed locally,
you can compile any listing directly:

```bash
nvcc -O3 -std=c++17 -I common/ ch04-the-map-pattern/04-vector-add.cu \
     -o vector-add
./vector-add
```

The `-I common/` flag tells nvcc where to find `common.h`. Local execution
gives you the fastest iteration and access to profilers like Nsight Systems
and Nsight Compute.

## Which stage to use when

| Question                                      | Stage         |
| --------------------------------------------- | ------------- |
| "Did I get the algorithm right?"              | 1 (cudaforces) |
| "How fast is my kernel?"                      | 2 (Colab) or 3 (local) |
| "Why is my kernel slow? Where's the bottleneck?" | 3 (local + Nsight) |
| "I want to follow along while reading."       | 1 + 2 covers most chapters |

## License

MIT — see [LICENSE](LICENSE).
