// Quadrotor tracking example

// This script is just to show how to use the library, the data for this example is not tuned for our Crazyflie demo. Check the firmware code for more details.

// Make sure in glob_opts.hpp:
// - NSTATES = 12, NINPUTS=4
// - NHORIZON = anything you want
// - NTOTAL = 301 if using reference trajectory from trajectory_data/
// - tinytype = float if you want to run on microcontrollers
// States: x (m), y, z, phi, theta, psi, dx, dy, dz, dphi, dtheta, dpsi
// phi, theta, psi are NOT Euler angles, they are Rodiguez parameters
// check this paper for more details: https://ieeexplore.ieee.org/document/9326337
// Inputs: u1, u2, u3, u4 (motor thrust 0-1, order from Crazyflie)

#include <iostream>
#include<unistd.h>
#include <tinympc/admm_cuda.cuh>
#include "problem_data/quadrotor_20hz_params.hpp"
#include "trajectory_data/quadrotor_20hz_y_axis_line.hpp"
// #include <cuda_runtime.h>

Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

TinyCache cache;
TinyWorkspace work;
TinySettings settings;
TinySolver solver{&settings, &cache, &work};

 typedef Matrix<tinytype, NINPUTS, NHORIZON-1> tiny_MatrixNuNhm1;
    typedef Matrix<tinytype, NSTATES, NHORIZON> tiny_MatrixNxNh;
    typedef Matrix<tinytype, NSTATES, 1> tiny_VectorNx;

using namespace Eigen;


int main()
{
    cache.rho = rho_value;
        cache.Kinf = Map<Matrix<tinytype, NINPUTS, NSTATES, RowMajor>>(Kinf_data);
        cache.Pinf = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Pinf_data);
        cache.Quu_inv = Map<Matrix<tinytype, NINPUTS, NINPUTS, RowMajor>>(Quu_inv_data);
        cache.AmBKt = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(AmBKt_data);

        work.Adyn = Map<Matrix<tinytype, NSTATES, NSTATES, RowMajor>>(Adyn_data);
        work.Bdyn = Map<Matrix<tinytype, NSTATES, NINPUTS, RowMajor>>(Bdyn_data);
        work.Q = Map<Matrix<tinytype, NSTATES, 1>>(Q_data);
        work.R = Map<Matrix<tinytype, NINPUTS, 1>>(R_data);

        // std::cout<<"cache->Kinf " << cache.Kinf<<std::endl;
 


        // TODO: Make function to handle variable initialization so specific important parts (like nx, nu, and N) aren't forgotten
        // work.nx = NSTATES;
        // work.nu = NINPUTS;
        // work.N = NHORIZON;

        work.u_min = tiny_MatrixNuNhm1::Constant(-0.5);
        work.u_max = tiny_MatrixNuNhm1::Constant(0.5);
        work.x_min = tiny_MatrixNxNh::Constant(-5);
        work.x_max = tiny_MatrixNxNh::Constant(5);

        work.Xref = tiny_MatrixNxNh::Zero();
        work.Uref = tiny_MatrixNuNhm1::Zero();

        work.x = tiny_MatrixNxNh::Zero();
        work.q = tiny_MatrixNxNh::Zero();
        work.p = tiny_MatrixNxNh::Zero();
        work.v = tiny_MatrixNxNh::Zero();
        work.vnew = tiny_MatrixNxNh::Zero();
        work.g = tiny_MatrixNxNh::Zero();

        work.u = tiny_MatrixNuNhm1::Zero();
        work.r = tiny_MatrixNuNhm1::Zero();
        work.d = tiny_MatrixNuNhm1::Zero();
        work.z = tiny_MatrixNuNhm1::Zero();
        work.znew = tiny_MatrixNuNhm1::Zero();
        work.y = tiny_MatrixNuNhm1::Zero();

        work.primal_residual_state = 0;
        work.primal_residual_input = 0;
        work.dual_residual_state = 0;
        work.dual_residual_input = 0;
        work.status = 0;
        work.iter = 0;


        settings.abs_pri_tol = 0.001;
        settings.abs_dua_tol = 0.001;
        settings.max_iter = 100;
        settings.check_termination = 1;
        settings.en_input_bound = 1;
        settings.en_state_bound = 1;

        tiny_VectorNx x0, x1; // current and next simulation states

        // Map data from trajectory_data
        Matrix<tinytype, NSTATES, NTOTAL> Xref_total = Eigen::Map<Matrix<tinytype, NSTATES, NTOTAL>>(Xref_data);
        work.Xref = Xref_total.block<NSTATES, NHORIZON>(0, 0);

        // Initial state
        x0 = work.Xref.col(0);

   
        for (int k = 0; k < 1; ++k)
        {
            // std::cout << "tracking error: " << (x0 - work.Xref.col(1)).norm() << std::endl;

            // 1. Update measurement
            // sleep(1);
            work.x.col(0) = x0;


            // 2. Update reference
            work.Xref = Xref_total.block<NSTATES, NHORIZON>(0, k);

            // std::cout<<"work.xref "<< work.Xref<<std::endl;


            // 3. Reset dual variables if needed
            work.y = tiny_MatrixNuNhm1::Zero();
            work.g = tiny_MatrixNxNh::Zero();

            // 4. Solve MPC problem
            // std::cout<<solver.cache;

            // std::cout<<"MAYBE mistake outer" << solver.cache->Kinf<<std::endl;

            tiny_solve_cpu(&solver);

            // hello();

            // std::cout << work.iter << std::endl;
            // std::cout << work.u.col(0).transpose().format(CleanFmt) << std::endl;

            // 5. Simulate forward
            x1 = work.Adyn * x0 + work.Bdyn * work.u.col(0);
            x0 = x1;

            // std::cout << x0.transpose().format(CleanFmt) << std::endl;
        }
    // }

    return 0;
}


