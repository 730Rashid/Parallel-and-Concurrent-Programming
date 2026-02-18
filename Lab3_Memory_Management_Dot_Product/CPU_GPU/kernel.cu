#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void dotProductShared(int* c, int* a, int* b)
{
    __shared__ int dataPerBlock[8];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    dataPerBlock[threadIdx.x] = a[i] * b[i];

    // Ensure all threads have written before Thread 0 reads
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int subtotal = 0;
        
        for (int k = 0; k < blockDim.x; k++)
        {
            subtotal += dataPerBlock[k];
        }
        c[blockIdx.x] = subtotal;
    }
}

void runConfig(int* a, int* b, int* c, int arraySize, int numBlocks, int threadsPerBlock)
{
    for (int i = 0; i < arraySize; i++)
    {
        c[i] = 0;
    }

    dotProductShared<<<numBlocks, threadsPerBlock>>>(c, a, b);

    cudaDeviceSynchronize();

    int dotProduct = 0;

    for (int i = 0; i < numBlocks; i++)
    {
        dotProduct += c[i];
    }

    printf("<<<%d, %d>>> -> %d subtotal(s), Dot Product = %d\n",
        numBlocks, threadsPerBlock, numBlocks, dotProduct);
}

int main()
{
    const int arraySize = 8;

    int* a, * b, * c;

    cudaMallocManaged(&a, arraySize * sizeof(int));
    cudaMallocManaged(&b, arraySize * sizeof(int));
    cudaMallocManaged(&c, arraySize * sizeof(int));

    for (int i = 0; i < arraySize; i++)
    {
        a[i] = i + 1;
        b[i] = (i + 1) * 10;
        c[i] = 0;
    }

    printf("Dot Product using Shared Memory \n\n");
    runConfig(a, b, c, arraySize, 1, 8);
    runConfig(a, b, c, arraySize, 2, 4);
    runConfig(a, b, c, arraySize, 4, 2);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
