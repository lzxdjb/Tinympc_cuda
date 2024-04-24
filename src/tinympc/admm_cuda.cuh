#pragma once



#include "types.hpp"

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    }
}

int tiny_solve_cuda(TinySolver *solver);


void hello();



#define NHORIZON 10
#define NUM_THREADS 256