# Chapter 14 — Warps and Occupancy

This chapter explains how the hardware actually runs your threads: warps of 32 executing in lockstep (SIMT), what happens when a branch splits a warp (divergence), and how registers, shared memory, and block size limit how many warps an SM can keep resident (occupancy).

Its code consists of short illustrative fragments (a divergent branch and its branchless rewrite, `threadIdx.x` versus `blockIdx.x` conditions, a block-size sweep loop, `__launch_bounds__`), not standalone listings, so there are no `.cu` files here. The kernels it analyzes are the ones you have already written: the reduction from [Chapter 5](../ch05-the-reduction-pattern/), the histogram from [Chapter 8](../ch08-the-histogram-pattern/), and the tiled kernels from [Chapters 11](../ch11-shared-memory/) and [12](../ch12-parallel-matrix-multiplication/).

Measuring divergence and occupancy needs a physical GPU (swforces.com's judge runs your kernels on CPUs); the chapter shows how to do it with Nsight Compute, for example in a free Google Colab GPU session.
