// #include "admm_kernel.cuh"
#include "admm_cuda.cuh"
#include "admm.hpp"

#include <iostream>

#include <cuda_runtime.h>
// #include "admm_kernel.cuh"



__global__ void solve_kernel(TinySolver *solver)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NHORIZON)
    {
        solver->work->y.col(i) += solver->work->u.col(i) - solver->work->znew.col(i);
        solver->work->g.col(i) += solver->work->x.col(i) - solver->work->vnew.col(i);
    }
}


#define checkCudaErrors(x) check((x), #x, __FILE__, __LINE__)

int tiny_solve_cuda(TinySolver *solver)
{
    // Initialize variables

    solver->work->status = 11; // TINY_UNSOLVED
    solver->work->iter = 0;

    TinySolver *solver_gpu;
    // printf("asdfasdf");

    checkCudaErrors(cudaMalloc(&solver_gpu, sizeof(TinySolver)));

  
    checkCudaErrors(cudaMemcpy(solver_gpu,solver ,sizeof(TinySolver), cudaMemcpyHostToDevice));

    for (int i = 0; i < solver->settings->max_iter; i++)
    {

        solve_kernel<<<1, 256>>>(solver_gpu);

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
    checkCudaErrors(cudaMemcpy(solver, solver_gpu, sizeof(TinySolver), cudaMemcpyDeviceToHost));
    return 1;
}

void hello()
{
    printf("hello world!");
}
