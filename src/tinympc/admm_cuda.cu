// #include "admm_kernel.cuh"
#include "admm_cuda.cuh"
#include "admm.hpp"

#include <iostream>

#include <cuda_runtime.h>
// #include "admm_kernel.cuh"



__global__ void solve_kernel(TinySolver *solver)
{    
    // printf("asdfasd");

    // int i = blockIdx.x * blockDim.x + threadIdx.x;
    solver->cache->rho = 0.51928472;
    solver->settings->max_iter= 10;
    // // printf(solver);
    // // TinyCache *cache_gpu = solver->cache;
    // cache_gpu->rho = 0.5;
    // solver->settings->max_iter = 1;

    // solver->settings = solver->settings;
    // solver->cache = solver->cache;
    // solver->work = solver->work;
    // solver->a = 1;
    // if (i < NHORIZON)
    // {
    //     solver->work->y.col(i) += solver->work->u.col(i) - solver->work->znew.col(i);
    //     solver->work->g.col(i) += solver->work->x.col(i) - solver->work->vnew.col(i);
    // }
}


#define checkCudaErrors(x) check((x), #x, __FILE__, __LINE__)

int tiny_solve_cuda(TinySolver *solver)
{
   
    solver->work->status = 1; // TINY_UNSOLVED
    solver->work->iter = 0;

    TinySolver *solver_gpu;
    checkCudaErrors(cudaMallocManaged((void**)&solver_gpu, sizeof(TinySolver)));

    TinyCache * device_cache;
    TinySettings *device_setting;
    TinyWorkspace * device_workspace;

    checkCudaErrors(cudaMalloc((void**)&device_cache, sizeof(TinyCache)));
    checkCudaErrors(cudaMemcpy(device_cache ,solver->cache ,sizeof(TinyCache), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&device_setting, sizeof(TinySettings)));
    checkCudaErrors(cudaMemcpy(device_setting ,solver->settings ,sizeof(TinySettings), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&device_workspace, sizeof(TinyWorkspace)));
    checkCudaErrors(cudaMemcpy(device_workspace ,solver->work ,sizeof(TinyWorkspace), cudaMemcpyHostToDevice));
    
    TinyCache * host_cache;
    TinySettings *host_setting;
    TinyWorkspace * host_workspace;

    host_cache = solver->cache;
    solver->cache = device_cache;

    host_setting = solver->settings;
    solver->settings = device_setting;

    host_workspace = solver->work;
    solver->work = device_workspace;

    checkCudaErrors(cudaMemcpy(solver_gpu ,solver ,sizeof(TinySolver), cudaMemcpyHostToDevice));

    solver->cache = host_cache;
    solver->settings = host_setting;
    solver->work = host_workspace;
    
    solve_kernel<<<1 , 32>>>(solver_gpu);

    TinyCache * debug_cache;
    checkCudaErrors(cudaMallocHost((void**)&debug_cache, sizeof(TinyCache)));
    checkCudaErrors(cudaMemcpy(debug_cache, solver_gpu->cache, sizeof(TinySolver), cudaMemcpyDeviceToHost));

    TinyCache * debug_setting;
    checkCudaErrors(cudaMallocHost((void**)&debug_setting, sizeof(TinySettings)));
    checkCudaErrors(cudaMemcpy(debug_setting, solver_gpu->settings, sizeof(TinySettings), cudaMemcpyDeviceToHost));

    TinyCache * debug_workspace;
    checkCudaErrors(cudaMallocHost((void**)&debug_workspace, sizeof(TinyWorkspace)));
    checkCudaErrors(cudaMemcpy(debug_workspace, solver_gpu->work, sizeof(TinyWorkspace), cudaMemcpyDeviceToHost));

    std::cout<<debug_cache->rho ;
    checkCudaErrors(cudaDeviceSynchronize());

    
 
   

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
