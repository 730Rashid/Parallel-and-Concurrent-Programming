# 500083 Lab Book

## Week 1 - Lab A

28 Jan 2026

### Q1. Hello World

**Question:**

Question 1: Set up CUDA Project in Visual Studio 2022
    - Setup necessary files inside VS

Question 2: Understanding the CUDA Programming Model
    Cuda has a very strict workflow that needs to be followed:
        - Create Data
        - Pass Data
        - Compute 
        - Retrieve Data

Question 3: Refactoring for Heterogeneous Understanding
    CUDA uses a function called addwithCuda() which does the complex things for us but we need to understand the Host and Device workflow manually.

Question 4: Error Handling and Synchronization
    - Understand CUDA functions to find out whether the script ran successfully or if it was a failure. Understand the best way to implement them.

Question 5: Performance Profiling
    - Understand how parallelism works in my code by changing values of variables and re-running the script. 



**Solution:**
{1,2,3,4,5} + {10,20,30,40,50} = {11,22,33,44,55}

int main()
{
    // 1. HOST: Initialize Data [cite: 117]
    const int arraySize = 50;
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

    cudaStatus = cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
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

    cudaStatus = cudaMemcpy(dev_c, c, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 3. DEVICE: Compute
    // TODO: Launch the kernel with 1 block and 5 threads [cite: 138]

    printf("Launching Kernel...\n");
    cudaEventRecord(start);

    addKernel<<<1, arraySize >>>(dev_c, dev_a, dev_b);

    cudaEventRecord(stop);

    // TODO: Wait for the GPU to finish [cite: 148]
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken: %f ms\n", milliseconds);

    // 4. DEVICE -> HOST: Copy Back
    // TODO: Copy the result 'dev_c' back to host array 'c' [cite: 149]
    // Hint: Use cudaMemcpyDeviceToHost
    
    cudaStatus = cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

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

I have learnt how to make a new CUDA solution in Visual Studio 2022. I followed the lab sheet on canvas which explains in detail what the pre-written code does and how it works. For example the __Global__ tells us that the funtion runs on the GPU but is called from the CPU. The function "addKernal" is where we have to specify how many blocks and threads we want. 

-- Cudamaloc() is a pre-built function that allocates memory on the GPU VRAM which is then processed by the kernals on the GPU. And i called the funtion for each pointer dev_a, dev_b, dev_c. 

-- cudaMemcpy() is also a pre-built function where it transfers data from the CPU to the GPU. I did this for all 3 variables so speed up compute as the GPU is much faster than the CPU.

-- cudaEvent() is also a pre-built function to monitor GPU's progress and accurately measure time.

-- cudeFree() is also a pre-built function to deallocate memory on the GPU that was used by cudaMalloc.

The opening of the CUDA file as I create the file is really good as you can understand how each part of the code works behind the scenes and how each function is worked and when to use them. The part I liked the most was the "Refactoring for Heterogeneous Understanding" where I was able to dive a little deeper and understand how Host and Device works and how to allocate them and when. I also liked how the CUDA workdlow is strict where I had to follow this "Create Data -> Pass Data -> Compute -> Retrieve Data". This has given me enough information to work with the second CUDA lab.




*Did you make any mistakes?*

I did have some issues when doing the fifth exercise where I had trouble pasted the code from the lab where we can record the execution time. it was pretty much a trial and error until I got it to work. I forgot to set the device using cudaSetDevice(0) which caused an error when trying to allocate memory on the GPU. 

I also had some trouble with copying the data back from the device to the host, I had to check the lab sheet again to make sure I was using cudaMemcpy with the correct parameters as well as the hints given.



*In what way has your knowledge improved?*

I now understand most of what the pre-coded CUDA script looks like and how most the functions work and when they're called. I know how to use addKernal() and specify the number threads and blocks inside <<< >>>. I also understand why CUDA is used and the purpose it serves. I also understand that CUDA follows a very strict workflow which is "Create Data -> Pass Data -> Compute -> Retrieve Data."




**Questions:**

No questions thank you.

### Q2. Console Window

![alt text](<Screenshot 2026-01-30 113250.png>)
...
