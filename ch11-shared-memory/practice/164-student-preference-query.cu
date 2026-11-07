// Chapter 11 — Practice on swforces
// Problem 164: Student Preference Query
// https://swforces.com/problem/164
//
// Reference solution (not printed in the book). The host encodes every
// answer as a small integer (subject, fruit, color each 0..2; a "-" in a
// query becomes -1, meaning "any"). On the device, the 3-Step Recipe:
// Load: the block copies the whole student table (n <= 10 rows of three
// ints) into shared memory once. Collaborate: thread q walks the table in
// shared memory and counts the students matching query q. Write back: each
// thread writes its own count. Every query reuses the same table, which is
// why it belongs on the whiteboard rather than being re-read from global
// memory by every thread.

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_STUDENTS 16

__global__ void queryKernel(int* students, int n, int* queries, int* counts, int m) {
    __shared__ int table[MAX_STUDENTS * 3];    // n rows of (subject, fruit, color)
    int tid = threadIdx.x;

    // Step 1: Load. The threads copy the table into shared memory cooperatively
    for (int i = tid; i < n * 3; i += blockDim.x) {
        table[i] = students[i];
    }
    __syncthreads();

    // Step 2: Collaborate. Each thread answers one query from the shared table
    if (tid < m) {
        int qs = queries[tid * 3], qf = queries[tid * 3 + 1], qc = queries[tid * 3 + 2];
        int count = 0;
        for (int i = 0; i < n; i++) {
            int match = (qs < 0 || table[i * 3]     == qs)
                     && (qf < 0 || table[i * 3 + 1] == qf)
                     && (qc < 0 || table[i * 3 + 2] == qc);
            if (match) count++;
        }
        // Step 3: Write back. One count per query
        counts[tid] = count;
    }
}

// Encode a word as its index in the category's list; "-" (or anything else) is -1
int encode(const char* word, const char* options[3]) {
    for (int i = 0; i < 3; i++) {
        if (strcmp(word, options[i]) == 0) return i;
    }
    return -1;
}

int main() {
    const char* subjects[3] = {"kor", "eng", "math"};
    const char* fruits[3]   = {"apple", "pear", "orange"};
    const char* colors[3]   = {"red", "blue", "green"};

    int n, m;
    scanf("%d %d", &n, &m);
    int h_students[MAX_STUDENTS * 3], h_queries[MAX_STUDENTS * 3], h_counts[MAX_STUDENTS];
    char a[16], b[16], c[16];
    for (int i = 0; i < n; i++) {
        scanf("%s %s %s", a, b, c);
        h_students[i * 3]     = encode(a, subjects);
        h_students[i * 3 + 1] = encode(b, fruits);
        h_students[i * 3 + 2] = encode(c, colors);
    }
    for (int i = 0; i < m; i++) {
        scanf("%s %s %s", a, b, c);
        h_queries[i * 3]     = encode(a, subjects);
        h_queries[i * 3 + 1] = encode(b, fruits);
        h_queries[i * 3 + 2] = encode(c, colors);
    }

    int *d_students, *d_queries, *d_counts;
    cudaMalloc(&d_students, n * 3 * sizeof(int));
    cudaMalloc(&d_queries, m * 3 * sizeof(int));
    cudaMalloc(&d_counts, m * sizeof(int));
    cudaMemcpy(d_students, h_students, n * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_queries, h_queries, m * 3 * sizeof(int), cudaMemcpyHostToDevice);

    queryKernel<<<1, 32>>>(d_students, n, d_queries, d_counts, m);

    cudaMemcpy(h_counts, d_counts, m * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; i++) printf("%d\n", h_counts[i]);

    cudaFree(d_students);
    cudaFree(d_queries);
    cudaFree(d_counts);
    return 0;
}
