#pragma once
#include<iostream>

__device__ static inline double  copy(double *a , double *b , int number)
{
    for (int i = 0 ; i < number ;i++)
    {
        a[i] = b[i];
    }
}


__device__ static inline double  dot_product(double *a , double *b , int number)
{
    double result = 0;
    for (int i = 0 ; i < number ;i++)
    {
        result += a[i] * b[i];
    }
    return result;
}

