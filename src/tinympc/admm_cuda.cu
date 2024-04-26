// #include "admm_kernel.cuh"
#include "admm_cuda.cuh"
#include "admm.hpp"
#include "admm_kernel.cuh"

#include <iostream>

#include <cuda_runtime.h>

#define TOTAL_SIZE 9
#define ITERATION 1
#define checkCudaErrors(x) check((x), #x, __FILE__, __LINE__)

clock_t start, end;
double cpu_time_used;


__global__ void solve_kernel(TinySolver *solver)
{
    int idx = threadIdx.x;

    double uCache[4] = {0};
    double xCache[12] = {0};

    for (int i = 0; i < 12; i++)
    {
        xCache[i] = solver->work->x.col(idx)[i];
    }

    double bigxCache[12][10] = {0};

    for (int j = 0 ; j < 12 ; j++)
    {
        for (int i = 0 ; i < 10 ; i++)
        {
            bigxCache[j][i] =  solver->work->x.row(j)[i];
        }
    }

  
    double KinfCache[4][12] = {0};
    double AdynCache[12][12] = {0};
    double BdynCache[12][4] = {0};

    for (int j = 0; j < 4; j++)
    {
        for (int i = 0; i < 12; i++)
        {
            KinfCache[j][i] = solver->cache->Kinf.row(j)[i];
        }
    }

    for (int j = 0; j < 12; j++)
    {
        for (int i = 0; i < 12; i++)
        {
            AdynCache[j][i] = solver->work->Adyn.row(j)[i];
        }
    }

    for (int j = 0; j < 12; j++)
    {
        for (int i = 0; i < 4; i++)
        {
            BdynCache[j][i] = solver->work->Bdyn.row(j)[i];
        }
    }

    double dCache[4] = {0};
    for (int i = 0; i < 4; i++)
    {
        dCache[i] = solver->work->d.col(idx)[i];
    }

    for (int i = 0; i < 4; i++)
    {
        uCache[i] = solver->work->u.col(idx)[i];
    }

    double temp_x[12] = {0};
    for (int k = 0 ; k < 12 ; k++)
    {
        temp_x[k] = bigxCache[k][idx];
    }

    // work
    for (int iteration = 0; iteration < ITERATION; iteration++)
    {
        for (int i = 0; i < 4; i++)
        {
            // uCache[i] = - temp[i] -
           
            uCache[i] = - dot_product(KinfCache[i], temp_x, 12) - dCache[i];
        }
        for (int i = 0; i < 12; i++)
        {
            // uCache[i] = - temp[i] -
            xCache[i] = dot_product(AdynCache[i], temp_x , 12) + dot_product(BdynCache[i] , uCache , 4);
        }

        __syncthreads();

        for (int k = 0 ; k < 12 ; k++)
        {
            bigxCache[k][idx + 1] = xCache[k];
        }

        __syncthreads();

    }


  

    ////work

    for (int i = 0; i < 4; i++)
    {
        solver->work->u.col(idx)[i] = uCache[i];
    }

    for (int j = 0; j < 12; j++)
    {
        for (int i = 0; i < 12; i++)
        {
            bigxCache[j][i] = solver->work->x.row(j)[i];
        }
    }
}




int tiny_solve_cuda(TinySolver *solver)
{

    solver->work->status = 1; // TINY_UNSOLVED
    solver->work->iter = 0;

    TinySolver *solver_gpu;
    checkCudaErrors(cudaMallocManaged((void **)&solver_gpu, sizeof(TinySolver)));

    TinyCache *device_cache;
    TinySettings *device_setting;
    TinyWorkspace *device_workspace;

    checkCudaErrors(cudaMalloc((void **)&device_cache, sizeof(TinyCache)));
    checkCudaErrors(cudaMemcpy(device_cache, solver->cache, sizeof(TinyCache), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&device_setting, sizeof(TinySettings)));
    checkCudaErrors(cudaMemcpy(device_setting, solver->settings, sizeof(TinySettings), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void **)&device_workspace, sizeof(TinyWorkspace)));
    checkCudaErrors(cudaMemcpy(device_workspace, solver->work, sizeof(TinyWorkspace), cudaMemcpyHostToDevice));

    TinyCache *host_cache;
    TinySettings *host_setting;
    TinyWorkspace *host_workspace;

    host_cache = solver->cache;
    solver->cache = device_cache;

    host_setting = solver->settings;
    solver->settings = device_setting;

    host_workspace = solver->work;
    solver->work = device_workspace;

    checkCudaErrors(cudaMemcpy(solver_gpu, solver, sizeof(TinySolver), cudaMemcpyHostToDevice));

    solver->cache = host_cache;
    solver->settings = host_setting;
    solver->work = host_workspace;

    start = clock(); // Record starting time

    solve_kernel<<<1, 9>>>(solver_gpu);

    end = clock(); // Record ending time
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("eigen cudaGPU time used: %f seconds\n", cpu_time_used);

    TinyCache *debug_cache;
    checkCudaErrors(cudaMallocHost((void **)&debug_cache, sizeof(TinyCache)));
    checkCudaErrors(cudaMemcpy(debug_cache, solver_gpu->cache, sizeof(TinySolver), cudaMemcpyDeviceToHost));

    TinyCache *debug_setting;
    checkCudaErrors(cudaMallocHost((void **)&debug_setting, sizeof(TinySettings)));
    checkCudaErrors(cudaMemcpy(debug_setting, solver_gpu->settings, sizeof(TinySettings), cudaMemcpyDeviceToHost));

    TinyWorkspace *debug_workspace;
    checkCudaErrors(cudaMallocHost((void **)&debug_workspace, sizeof(TinyWorkspace)));
    checkCudaErrors(cudaMemcpy(debug_workspace, solver_gpu->work, sizeof(TinyWorkspace), cudaMemcpyDeviceToHost));

    std::cout << "cuda_version = " << debug_workspace->x << std::endl;
    checkCudaErrors(cudaDeviceSynchronize());

    start = clock(); // Record starting time

    for (int u = 0; u < ITERATION; u++)
    {
        forward_pass(solver);
    }

    end = clock(); // Record ending time

    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("eigen CPU time used: %f seconds\n", cpu_time_used);

    std::cout << "orginal = " << solver->work->x << std::endl;

    exit(0);

    for (int i = 0; i < solver->settings->max_iter; i++)
    {

        // Solve linear system with Riccati and roll out to get new trajectory
        forward_pass(solver);

        // Project slack variables into feasible domain
        update_slack(solver);

        // Compute next iteration of dual variables
        update_dual(solver);

        // Update linear control cost terms using reference trajectory, duals, and slack variables
        update_linear_cost(solver);

        // Check for whether cost is ~minimized~ by calculating residuals
        if (termination_condition(solver))
        {
            solver->work->status = 1; // TINY_SOLVED
            return 0;
        }

        // Save previous slack variables
        solver->work->v = solver->work->vnew;
        solver->work->z = solver->work->znew;

        backward_pass_grad(solver);

        solver->work->iter = i + 1;

        // std::cout << solver->work->primal_residual_state << std::endl;
        // std::cout << solver->work->dual_residual_state << std::endl;
        // std::cout << solver->work->primal_residual_input << std::endl;
        // std::cout << solver->work->dual_residual_input << "\n" << std::endl;
    }

    return 1;
}

void hello()
{
    printf("hello world!");
}
