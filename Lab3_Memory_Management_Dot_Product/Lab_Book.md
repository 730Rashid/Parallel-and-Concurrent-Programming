# 500083 Lab Book

## Week 3 - Lab 3 (Memory Management & Dot Product)

17 Feb 2026

---

### Exercise 1: Dot Product — CPU Baseline & CPU+GPU Solution

---

#### Exercise 1.1: CPU-Only Dot Product (Baseline)

**Question:**

Write a standard C++ CPU-only program to compute the dot product of vectors A and B. This acts as a "Gold Standard" to verify GPU results later.

Dot product formula: `result = A[0]*B[0] + A[1]*B[1] + ... + A[n-1]*B[n-1]`

**Solution:**

Created as a separate C++ Console Application project (`main.cpp`) in Visual Studio no CUDA required.

```cpp
int main() {
    const int N = 5;
    
    int a[N] = { 1, 2, 3, 4, 5 };
    int b[N] = { 10, 20, 30, 40, 50 };

    int result = 0;

    for (int i = 0; i < N; i++)
    {
        result += a[i] * b[i];
    }

    printf("CPU Dot Product = %d\n", result);
    return 0;
}
```

**Test data:**

A = {1, 2, 3, 4, 5}, B = {10, 20, 30, 40, 50}

Expected: (1×10) + (2×20) + (3×30) + (4×40) + (5×50) = 10 + 40 + 90 + 160 + 250 = **550**

**Sample output:**

```
CPU Dot Product = 550
```

---

#### Exercise 1.2: CPU+GPU Dot Product

**Question:**

Accelerate the dot product using CUDA by splitting the work into two steps:
1. **GPU**: Compute `C[i] = A[i] * B[i]` in parallel — one thread per element
2. **CPU**: Sum all elements of C to produce the final dot product

**Solution:**

```cpp
__global__ void multiplyKernel(int* c, const int* a, const int* b) {
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}
```

Kernel launch — 1 block, N threads (one per element):

```cpp
multiplyKernel<<<1, N>>>(dev_c, dev_a, dev_b);
```

After copying C back to host, the CPU sums the result:

```cpp
int dotProduct = 0;
for (int i = 0; i < N; i++) {
    dotProduct += c[i];
}
```

**Test data:**

A = {1, 2, 3, 4, 5}, B = {10, 20, 30, 40, 50}

C after GPU multiply = {10, 40, 90, 160, 250}

Dot product (CPU sum) = **550**

**Sample output:**

```
Launching multiplyKernel...
GPU+CPU Dot Product = 550
```

---

**Reflection:**

Starting with a CPU-only solution before touching CUDA was a useful approach — it gave me a clear answer to aim for before any GPU code was written. The sequential loop is simple but slow by nature, as each multiplication has to wait for the previous one to finish. Having this "Gold Standard" made it easy to confirm the GPU result was correct once Exercise 1.2 was up and running.

For the CUDA version, I split the problem into two parts: the GPU handles the per-element multiplications in parallel, and the CPU sums the results afterwards. The CUDA workflow was the same as Lab 1 — `cudaMalloc` to allocate memory on the device, `cudaMemcpy` to transfer data across, then launching the kernel before copying the result back. The key difference here was the kernel itself, which uses `threadIdx.x` to assign each thread its own element to process, so all five multiplications happen simultaneously rather than one after another.

Both methods produced **550**, which confirmed the GPU was computing correctly. It also highlighted where the GPU adds value — the multiplication step — whilst the summation stays on the CPU for now. Exercise 2 will move that final sum onto the GPU as well using shared memory.

**Questions:**

None.
