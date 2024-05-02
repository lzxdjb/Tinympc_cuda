// #include "admm_kernel.cuh"
#include "admm_cuda.cuh"
#include "admm.hpp"
#include "admm_kernel.cuh"

#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>

#define TOTAL_SIZE 9
#define ITERATION 100
#define checkCudaErrors(x) check((x), #x, __FILE__, __LINE__)

#define outerITERATION 300

clock_t start, end;
double cpu_time_used;

__global__ void solve_kernel(TinySolver *solver)
{
    //////argument

    int idx = threadIdx.x;

    double KinfCache[12] = {0};
    double AdynCache[12] = {0};
    double BdynCache[4] = {0};

    double u_min[9] = {0};
    double u_max[9] = {0};

    double x_min[10] = {0};
    double x_max[10] = {0};

    // update_linear_cost(4 , 9)

    __shared__ double rCache[4][9];
    double UrefCache[9] = {0};
    double R = 0;
    double rho = solver->cache->rho;

    // update_linear_cost(12 , 10) ;

    double qCache[10] = {0};
    double XrefCache[10] = {0};
    double Q = 0;

    __shared__ double pCache[12][10];
    __shared__ double XrefCache_colunm[12];
    __shared__ double PinfCache[12][12];

    // termination_condition
    int iter = solver->work->iter;
    int check_termination = solver->settings->check_termination;

    __shared__ double primal_residual_state[12], dual_residual_state[12], primal_residual_input[4], dual_residual_input[4];

    double temp_prs, temp_drs, temp_pri, temp_dri;

    double vCache[10] = {0};
    double zCache[9] = {0};

    double abs_pri_tol = solver->settings->abs_pri_tol;

    double abs_dua_tol = solver->settings->abs_dua_tol;

    /// backward_pass_grad

    double Quu_inv[4] = {0};
    __shared__ double Bdyn_colunm[4][12]; // need to optimize!
    double AmBKt[12] = {0};
    double Kinf_colunm[4] = {0};

    //////argument

    //////(initialize)

    if (idx < 4)
    {
        for (int i = 0; i < 12; i++)
        {
            KinfCache[i] = solver->cache->Kinf.row(idx)[i];
        }

        for (int i = 0; i < 9; i++)
        {
            u_min[i] = solver->work->u_min.row(idx)[i];
        }

        for (int i = 0; i < 9; i++)
        {
            u_max[i] = solver->work->u_max.row(idx)[i];
        }

        // update_linear_cost
        for (int i = 0; i < 9; i++)
        {
            UrefCache[i] = solver->work->Uref.row(idx)[i];
        }

        R = solver->work->R.row(idx)[0];
    }

    // forward_pass(19)

    for (int i = 0; i < 12; i++)
    {
        AdynCache[i] = solver->work->Adyn.row(idx)[i];
    }

    for (int i = 0; i < 4; i++)
    {
        BdynCache[i] = solver->work->Bdyn.row(idx)[i];
    }

    for (int i = 0; i < 10; i++)
    {
        x_max[i] = solver->work->x_max.row(idx)[i];
    }

    for (int i = 0; i < 10; i++)
    {
        x_min[i] = solver->work->x_min.row(idx)[i];
    }

    // update_linear_cost
    for (int i = 0; i < 10; i++)
    {
        qCache[i] = solver->work->Q.row(idx)[i];
    }

    for (int i = 0; i < 10; i++)
    {
        XrefCache[i] = solver->work->Xref.row(idx)[i];
    }

    Q = solver->work->Q.row(idx)[0];

    // test
    // if(idx == 0)
    {
        for (int i = 0; i < 12; i++)
        {
            XrefCache_colunm[i] = solver->work->Xref.col(NHORIZON - 1)[i];
        }

        // test
        for (int i = 0; i < 12; i++)
        {
            for (int j = 0; j < 12; j++)
            {
                PinfCache[i][j] = solver->cache->Pinf.row(i)[j];
            }
        }
    }

    // termination_condition

    for (int i = 0; i < 10; i++)
    {
        vCache[i] = solver->work->v.row(idx)[i];
    }

    for (int i = 0; i < 9; i++)
    {
        zCache[i] = solver->work->z.row(idx)[i];
    }

    /// backward_pass_grad

    if (idx < 4)
    {
        test temp = solver->work->Bdyn.transpose();
        for (int i = 0; i < 12; i++)
        {
            Bdyn_colunm[idx][i] = temp.row(idx)[i];
        }

        for (int j = 0; j < 4; j++)
        {
            Quu_inv[j] = solver->cache->Quu_inv.row(idx)[j];
        }
    }

    for (int i = 0; i < 12; i++)
    {
        AmBKt[i] = solver->cache->AmBKt.row(idx)[i];
    }

    for (int i = 0; i < 4; i++)
    {
        Kinf_colunm[i] = solver->cache->Kinf.col(idx)[i];
    }

    //////(initialize)

    /// somthing should put in forward_pass
    __shared__ double uCache[4][9];
    __shared__ double xCache[12][10];
    __shared__ double dCache[4][9];

    if (idx < 4)
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                uCache[i][j] = solver->work->u.row(i)[j];
            }
        }

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                dCache[i][j] = solver->work->d.row(i)[j];
            }
        }
    }

    for (int i = 0; i < 12; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            xCache[i][j] = solver->work->x.row(i)[j];
        }
    }

    double temp_x[12] = {0}; // for p
    double temp_u[9] = {0};
    double temp_d[9] = {0};

    double znew_cache[9] = {0};
    double y_cache[9] = {0};
    double vnew_cache[10] = {0};
    double g_cache[10] = {0};

    for (int i = 0; i < 9; i++)
    {
        y_cache[i] = solver->work->y.row(idx)(i);
    }

    for (int i = 0; i < 10; i++)
    {
        g_cache[i] = solver->work->g.row(idx)(i);
    }

    int en_input_bound = solver->settings->en_input_bound;

    int en_state_bound = solver->settings->en_state_bound;

    __syncthreads();

    ////////// workspace
    for (int outer = 0; outer < outerITERATION; outer++)
    {

        for (int iteration = 0; iteration < ITERATION; iteration++)
        {

            // forword_pass
            for (int i = 0; i < 9; i++)
            {

                if (idx < 4)
                {
                    for (int j = 0; j < 12; j++)
                    {
                        temp_x[j] = xCache[j][i];
                    }

                    uCache[idx][i] = -dot_product(KinfCache, temp_x, 12) - dCache[idx][i];
                }

                __syncthreads();

                for (int j = 0; j < 4; j++)
                {
                    temp_u[j] = uCache[j][i];
                }

                for (int j = 0; j < 12; j++)
                {
                    temp_x[j] = xCache[j][i];
                }

                xCache[idx][i + 1] = dot_product(AdynCache, temp_x, 12) + dot_product(BdynCache, temp_u, 4);

                // if(idx == 0)
                // {
                //     for(int i = 0 ; i < 12 ; i ++)
                //     {
                //         for(int j = 0 ; j < 10 ; j ++)
                //         {
                //             printf("%f = " , xCache[i][j]);
                //         }
                //     }

                // }

                __syncthreads();
            }
            // break;
            // update_slack(4)
            if (idx < 4)
            {
                for (int i = 0; i < 9; i++)
                {
                    temp_u[i] = uCache[idx][i];
                }

                for (int i = 0; i < 9; i++)
                {
                    znew_cache[i] = temp_u[i] + y_cache[i];
                }

                // Box constraints on input
                if (en_input_bound)
                {
                    for (int i = 0; i < 9; i++)
                    {
                        znew_cache[i] = min(u_max[i], max(u_min[i], znew_cache[i]));
                    }
                }

                // update_dual
                for (int i = 0; i < 9; i++)
                {
                    y_cache[i] += temp_u[i] - znew_cache[i];
                }

                // update_linear_cost

                for (int i = 0; i < 9; i++)
                {
                    rCache[idx][i] = -UrefCache[i] * R;
                }

                for (int i = 0; i < 9; i++)
                {
                    rCache[idx][i] -= rho * (znew_cache[i] - y_cache[i]);
                }
            }

            // update_slack (10)
            for (int i = 0; i < 10; i++)
            {
                temp_x[i] = xCache[idx][i];
            }

            for (int i = 0; i < 10; i++)
            {
                vnew_cache[i] = temp_x[i] + g_cache[i];
            }

            // Box constraints on state
            if (en_state_bound)
            {
                for (int i = 0; i < 10; i++)
                {
                    vnew_cache[i] = min(x_max[i], max(x_min[i], vnew_cache[i]));
                }
            }

            // update_dual
            for (int i = 0; i < 10; i++)
            {
                g_cache[i] += temp_x[i] - vnew_cache[i];
            }

            //  update_linear_cost
            for (int i = 0; i < 10; i++)
            {
                qCache[i] = -(XrefCache[i] * Q);
            }
            for (int i = 0; i < 10; i++)
            {
                qCache[i] -= rho * (vnew_cache[i] - g_cache[i]);
            }

            for (int i = 0; i < 12; i++)
            {
                temp_x[i] = PinfCache[i][idx];
            }

            pCache[idx][NHORIZON - 1] = -dot_product(XrefCache_colunm, temp_x, 12);

            pCache[idx][NHORIZON - 1] -= rho * (vnew_cache[NHORIZON - 1] - g_cache[NHORIZON - 1]);

            // termination_condition
            for (int i = 0; i < 10; i++)
            {
                temp_x[i] = xCache[idx][i];
            }
            if (iter % check_termination == 0)
            {
                primal_residual_state[idx] = cwiseAbs_maxCoeff(temp_x, vnew_cache, 10);
                dual_residual_state[idx] = cwiseAbs_maxCoeff(vCache, vnew_cache, 10);

                // termination_condition
                if (idx < 4)
                {
                    primal_residual_input[idx] = cwiseAbs_maxCoeff(temp_u, znew_cache, 9);

                    dual_residual_input[idx] = cwiseAbs_maxCoeff(zCache, znew_cache, 9);
                }

                __syncthreads();

                temp_prs = findmax(primal_residual_state, 12);

                temp_drs = findmax(dual_residual_state, 12) * rho;

                if (idx < 4)
                {
                    temp_pri = findmax(primal_residual_input, 4);

                    temp_dri = findmax(dual_residual_input, 4) * rho;
                }

                __syncthreads();

                if (temp_prs < abs_pri_tol && temp_pri < abs_pri_tol && temp_drs < abs_pri_tol && temp_dri < abs_dua_tol && temp_drs < abs_dua_tol) // I do not check that.
                {
                    // break; // Do not do anything
                }
            }

            // Save previous slack variables
            for (int i = 0; i < 10; i++)
            {
                vCache[i] = vnew_cache[i];
            }
            if (idx < 4)
            {
                for (int j = 0; j < 9; j++)
                {
                    zCache[j] = znew_cache[j];
                }
            }

            // backward_pass_grad
            for (int k = 8; k >= 0; k--)
            {

                if (idx < 4)
                {

                    for (int i = 0; i < 12; i++)
                    {
                        temp_x[i] = pCache[i][k + 1];
                    }

                    for (int i = 0; i < 4; i++)
                    {
                        double t = 0;
                        for (int j = 0; j < 12; j++)
                        {
                            t += Bdyn_colunm[i][j] * temp_x[j];
                        }
                        temp_d[i] = t;
                    }

                    for (int i = 0; i < 4; i++)
                    {
                        temp_d[i] = temp_d[i] + rCache[i][k];
                    }

                    double temp = 0;
                    for (int j = 0; j < 4; j++)
                    {
                        temp += Quu_inv[j] * temp_d[j];
                    }

                    dCache[idx][k] = temp;
                }

                __syncthreads();

                for (int i = 0; i < 12; i++)
                {
                    temp_x[i] = pCache[i][k + 1];
                }

                for (int i = 0; i < 4; i++)
                {
                    temp_d[i] = rCache[i][k];
                }

                pCache[idx][k] = qCache[k] + dot_product(AmBKt, temp_x, 12) - dot_product(Kinf_colunm, temp_d, 4);

                __syncthreads();
            }
        }
    }

    ////////// workspace

    ////////// load

    // if (idx < 4)
    // {
    //     for (int i = 0; i < 9; i++)
    //     {
    //         solver->work->u.row(idx)[i] = uCache[idx][i];
    //     }

    //     for (int i = 0; i < 9; i++)
    //     {
    //         solver->work->znew.row(idx)[i] = znew_cache[i];
    //     }

    //     for (int i = 0; i < 9; i++)
    //     {
    //         solver->work->y.row(idx)[i] = y_cache[i];
    //     }

    //     for (int i = 0; i < 9; i++)
    //     {
    //         solver->work->r.row(idx)[i] = rCache[idx][i];
    //     }
    // }

    for (int j = 0; j < 10; j++)
    {
        solver->work->x.row(idx)[j] = xCache[idx][j];
    }

    //     for (int j = 0; j < 10; j++)
    //     {
    //         solver->work->vnew.row(idx)[j] = vnew_cache[j];
    //     }

    //     for (int i = 0; i < 10; i++)
    //     {
    //         solver->work->g.row(idx)[i] = g_cache[i];
    //     }

    //     for (int i = 0; i < 10; i++)
    //     {
    //         solver->work->q.row(idx)[i] = qCache[i];
    //     }

    //     for (int i = 0; i < 10; i++)
    //     {
    //         solver->work->p.row(idx)[i] = pCache[idx][i];
    //     }

    //     // termination_condition:
    //     solver->work->primal_residual_state = temp_prs;

    //     solver->work->dual_residual_state = temp_drs;
    //     solver->work->primal_residual_input = temp_pri;

    //     solver->work->dual_residual_input = temp_dri;

    //     // Save previous slack variables

    //     for (int i = 0; i < 10; i++)
    //     {
    //         solver->work->v.row(idx)[i] = vCache[i];
    //     }

    //     if (idx < 4)
    //     {
    //         for (int i = 0; i < 9; i++)
    //         {
    //             solver->work->z.row(idx)[i] = zCache[i];
    //         }

    // //backward_pass_grad
    //         for (int i = 0; i < 4; i++)
    //         {
    //             for (int j = 0; j < 9; j++)
    //             {
    //                 solver->work->d.row(i)[j] = dCache[i][j];
    //             }
    //         }
    //     }

    //     for (int i = 0 ; i < 10 ; i++)
    //     {
    //         solver->work->p.row(idx)[i] = pCache[idx][i];
    //     }

    __syncthreads();
}
int tiny_solve_cuda(TinySolver *solver)
{
    // std::cout<<"MAYBE mistake 444444" << solver->cache->Kinf<<std::endl;

    solver->work->status = 11; // TINY_UNSOLVED
    solver->work->iter = 0;

    TinySolver *solver_gpu;
    checkCudaErrors(cudaMallocManaged((void **)&solver_gpu, sizeof(TinySolver)));

    TinyCache *device_cache;
    TinySettings *device_setting;
    TinyWorkspace *device_workspace;

    // std::cout<<"MAYBE mistake 111111" << solver->cache->Kinf<<std::endl;

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

    cpu_time_used = 0;

    cudaEvent_t start1;
    cudaEventCreate(&start1);
    cudaEvent_t stop1;
    cudaEventCreate(&stop1);
    cudaEventRecord(start1, NULL);
    start = clock(); // Record starting time

    // for (int i = 0; i < outerITERATION; i++)
    // {
    solve_kernel<<<1, 12>>>(solver_gpu);
    // }

    cudaEventRecord(stop1, NULL);
    cudaEventSynchronize(stop1);
    float msecTotal1 = 0.0f;
    cudaEventElapsedTime(&msecTotal1, start1, stop1);

    msecTotal1 = msecTotal1 / 1e+3;
    printf("eigen GPU time used: %f seconds\n", msecTotal1);

    TinyCache *debug_cache;
    checkCudaErrors(cudaMallocHost((void **)&debug_cache, sizeof(TinyCache)));
    checkCudaErrors(cudaMemcpy(debug_cache, solver_gpu->cache, sizeof(TinySolver), cudaMemcpyDeviceToHost));

    TinyCache *debug_setting;
    checkCudaErrors(cudaMallocHost((void **)&debug_setting, sizeof(TinySettings)));
    checkCudaErrors(cudaMemcpy(debug_setting, solver_gpu->settings, sizeof(TinySettings), cudaMemcpyDeviceToHost));

    TinyWorkspace *debug_workspace;
    checkCudaErrors(cudaMallocHost((void **)&debug_workspace, sizeof(TinyWorkspace)));
    checkCudaErrors(cudaMemcpy(debug_workspace, solver_gpu->work, sizeof(TinyWorkspace), cudaMemcpyDeviceToHost));

    // std::cout << "cuda_version = \n \n"
    //           << debug_workspace->x << std::endl;
    checkCudaErrors(cudaDeviceSynchronize());

    return 1;
}

int tiny_solve_cpu(TinySolver *solver)
{
    start = clock();

    for (int j = 0; j < outerITERATION; j++)
    {
        for (int k = 0; k < ITERATION; k++)
        {
            forward_pass(solver);
            update_slack(solver);
            update_dual(solver);
            update_linear_cost(solver);
            termination_condition(solver);
            solver->work->v = solver->work->vnew;
            solver->work->z = solver->work->znew;
            backward_pass_grad(solver);
        }
    }

    end = clock(); // Record ending time
    cpu_time_used += ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("eigen CPU time used: %f seconds\n", cpu_time_used);
}
