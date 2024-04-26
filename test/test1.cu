#include <time.h>
#include <Eigen.h>
#include <cuda_runtime.h>
#include <iostream>

using Eigen::Matrix;

typedef Matrix<double, 12, 10> tiny_VectorNu;
typedef Matrix<double, 4, 10> tiny_VectorNw;

tiny_VectorNu a;
tiny_VectorNu b;

tiny_VectorNw c;
tiny_VectorNw d;
__global__ void solve_kernel(tiny_VectorNu *a, tiny_VectorNu *b, tiny_VectorNw *c, tiny_VectorNw *d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double aa[10] = {0};
    double ba[10] = {0};

    for (int i = 0; i < 10; ++i)
    {
        aa[i] = a->row(idx)[i];
    }
    for (int i = 0; i < 10; ++i)
    {
        ba[i] = a->row(idx)[i];
    }

    for (int i = 0; i < 10000000; i++)
    {
        // int j =  + b->col(idx)[0];
        for (int i = 0; i < 10; i++)
        {
            aa[i] = aa[i] + ba[i];
        }

        //    a->col(idx)[0] = aa;
        // for (int i = 0; i < 10; ++i) {
        //     a->row(idx)[i] = aa[i];
        // }
        __syncthreads();

        if (idx < 4)
        {
            for (int i = 0; i < 10; ++i)
            {
                aa[i] = aa[i];
            }
            for (int i = 0; i < 10; ++i)
            {
                ba[i] = ba[i];
            }

            for (int i = 0; i < 10000000; i++)
            {
                // int j =  + b->col(idx)[0];
                for (int i = 0; i < 10; i++)
                {
                    aa[i] = aa[i] + ba[i];
                }
            }
        }

        for (int i = 0; i < 10; ++i)
        {
            c->row(idx)[i] = aa[i];
        }
    }
}

int main()
{
    clock_t start, end;
    double cpu_time_used;

    tiny_VectorNu *cuda_a;
    tiny_VectorNu *cuda_b;
    tiny_VectorNw *cuda_c;
    tiny_VectorNw *cuda_d;

    a.setZero();
    b.setZero();
    c.setZero();
    d.setZero();

    start = clock(); // Record starting time

    cudaMalloc((void **)&cuda_a, sizeof(tiny_VectorNu));
    cudaMemcpy(cuda_a, &a, sizeof(tiny_VectorNu), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&cuda_b, sizeof(tiny_VectorNu));
    cudaMemcpy(cuda_b, &b, sizeof(tiny_VectorNu), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&cuda_c, sizeof(tiny_VectorNw));
    cudaMemcpy(cuda_c, &c, sizeof(tiny_VectorNw), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&cuda_d, sizeof(tiny_VectorNw));
    cudaMemcpy(cuda_d, &d, sizeof(tiny_VectorNw), cudaMemcpyHostToDevice);

    // Perform some task for which you want to measure time
    // for (int i = 0; i < 1000000; i++) {
    // Do something

    solve_kernel<<<12, 1>>>(cuda_a, cuda_b, cuda_c, cuda_d);

    // }

    end = clock(); // Record ending time

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("eigen cudaGPU time used: %f seconds\n", cpu_time_used);

    start = clock(); // Record starting time

    // Perform some task for which you want to measure time
    for (int i = 0; i < 10000000; i++)
    {
        // Do something
        a = a + b;
        c = c + d;
    }

    end = clock(); // Record ending time

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("eigen CPU time used: %f seconds\n", cpu_time_used);
}