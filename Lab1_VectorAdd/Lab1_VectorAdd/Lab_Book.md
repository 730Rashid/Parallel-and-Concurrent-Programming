# 500083 Lab Book

## Week 1 - Lab A

28 Jan 2026

### Q1. Hello World

**Question:**

Exercise 1: Set up CUDA Project in Visual Studio 2022

Exercise 2: Understanding the CUDA Programming Model

Exercise 3: Refactoring for Heterogeneous Understanding

Exercise 4: Error Handling and Synchronization

Exercise 5: Performance Profiling



**Solution:**
{1,2,3,4,5} + {10,20,30,40,50} = {11,22,33,44,55}

int main()
{
    // 1. HOST: Initialize Data [cite: 117]
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Declare pointers for GPU (Device) memory [cite: 126-128]

    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;


    // 2. HOST -> DEVICE: Allocate and Copy
    // TODO: Allocate memory on the GPU for a, b, and c using cudaMalloc [cite: 129]
    // Hint: cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, arraySize  * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // TODO: Copy input vectors 'a' and 'b' from Host to Device using cudaMemcpy [cite: 132]
    // Hint: cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaStatus = cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // 3. DEVICE: Compute
    // TODO: Launch the kernel with 1 block and 5 threads [cite: 138]

    printf("Launching Kernel...\n");
    addKernel<<<1, arraySize >>>(dev_c, dev_a, dev_b);

    // TODO: Wait for the GPU to finish [cite: 148]
    cudaDeviceSynchronize();

    // 4. DEVICE -> HOST: Copy Back
    // TODO: Copy the result 'dev_c' back to host array 'c' [cite: 149]
    // Hint: Use cudaMemcpyDeviceToHost
    // Cleanup: Free GPU memory [cite: 151-153]

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);


    // Verify Result
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);
    return 0;


}

**Test data:**
arraySize = 1000
{1,2,3,4,5} + {10,20,30,40,50} = {11,22,33,44,55},

arraySize = 50
{1,2,3,4,5} + {10,20,30,40,50} = {11,22,33,44,55}



**Sample output:**

Launching Kernel...
Time taken: 0.681696 ms
{1,2,3,4,5} + {10,20,30,40,50} = {11,22,33,44,55}


Launching Kernel...
Time taken: 16.411648 ms
{1,2,3,4,5} + {10,20,30,40,50} = {11,22,33,44,55}


**Reflection:**

*Reflect on what you have learnt from this exercise.*
I have learnt how to make a new CUDA solution in Visual Studio 2022. I followed the lab sheet on canvas which explains in detail what the pre-written code does and how it works. For example the __Global__ tells us that the funtion runs on the GPU but is called from the CPU. The function "addKernal" is where we have to specify how many blocks
and threads we want.

*Did you make any mistakes?*

I did have some issues when doing the fifth excercise where I had trouble pasted the code from the lab where we can record the execution time. it was pretty much a trial and error until I got it to work. I forgot to set the device using cudaSetDevice(0) which caused an error when trying to allocate memory on the GPU. 
After adding that line the code worked properly. I also had some trouble with copying the data back from the device to the host, I had to check the lab sheet again to make sure I was using cudaMemcpy with the correct parameters aswell as the hints given.


**Questions:**

No questions thank you.

### Q2. Console Window

...
