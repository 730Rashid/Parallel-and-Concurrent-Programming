#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// Exercise 1.2 GPU Kernel - Element-wise Multiplication
// Each thread computes C[i] = A[i] * B[i]

__global__ void multiplyKernel(int* c, const int* a, const int* b) 
{
    int i = threadIdx.x;
    c[i] = a[i] * b[i];
}

int main() {

    const int N = 5;

    int a[N] = { 1, 2, 3, 4, 5 };
    int b[N] = { 10, 20, 30, 40, 50 };
    int c[N] = { 0 };

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    cudaError_t cudaStatus;

    // Choose GPU
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!\n");
        goto Error;
    }

    // 2. HOST -> DEVICE: Allocate GPU memory
    cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMalloc dev_a failed!\n"); 
        goto Error; 
    }

    cudaStatus = cudaMalloc((void**)&dev_b, N * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMalloc dev_b failed!\n"); 
        goto Error; 
    }

    cudaStatus = cudaMalloc((void**)&dev_c, N * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMalloc dev_c failed!\n"); 
        goto Error; 
    }

    // Copy A and B to GPU

    cudaStatus = cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMemcpy dev_a failed!\n"); 
        goto Error; 
    }

    cudaStatus = cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMemcpy dev_b failed!\n"); 
        goto Error; 
    }

    // 3. DEVICE: Compute - 1 block, N threads (one per element)

    printf("Running multiplyKernel...\n");
    multiplyKernel<<<1, N>>>(dev_c, dev_a, dev_b);

    cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceSynchronise failed!\n"); 
        goto Error; 
    }

    // 4. DEVICE -> HOST: Copy result C back
    cudaStatus = cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess) 
    { 
        fprintf(stderr, "cudaMemcpy c failed!\n"); 
        goto Error; 
    }

    // CPU Summation: sum all elements of C to get dot product
    {
        int dotProduct = 0;
        for (int i = 0; i < N; i++) {
            dotProduct += c[i];
        }
        // Compare this output with Exercise 1.1 CPU result to verify correctness
        printf("GPU+CPU Dot Product = %d\n", dotProduct);
    }

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}
