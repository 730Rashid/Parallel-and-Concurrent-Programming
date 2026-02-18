#include <stdio.h>

// Exercise 1.1: CPU-Only Dot Product 

int main() {
    
    const int N = 50;
    
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
