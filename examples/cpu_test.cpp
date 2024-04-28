
#include <iostream>
#include <time.h>
#include <tinympc/admm_cuda.cuh>
#include "problem_data/quadrotor_20hz_params.hpp"
#include "trajectory_data/quadrotor_20hz_y_axis_line.hpp"
// #include <cuda_runtime.h>

// #include <tinympc/admm.hpp>

Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

TinyCache cache;
TinyWorkspace work;
TinySettings settings;
TinySolver solver{&settings, &cache, &work};

int main()
{
    // Map data from problem_data (array in row-major order)
    cache.rho = rho_value;
    cache.Kinf = Eigen::Map<Matrix<tinytype, NINPUTS, NSTATES, Eigen::RowMajor>>(Kinf_data);
    cache.Pinf = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Pinf_data);
    cache.Quu_inv = Eigen::Map<Matrix<tinytype, NINPUTS, NINPUTS, Eigen::RowMajor>>(Quu_inv_data);
    cache.AmBKt = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(AmBKt_data);

    work.Adyn = Eigen::Map<Matrix<tinytype, NSTATES, NSTATES, Eigen::RowMajor>>(Adyn_data);
    work.Bdyn = Eigen::Map<Matrix<tinytype, NSTATES, NINPUTS, Eigen::RowMajor>>(Bdyn_data);
    work.Q = Eigen::Map<tiny_VectorNx>(Q_data);
    work.R = Eigen::Map<tiny_VectorNu>(R_data);
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

    clock_t start, end;
    double cpu_time_used = 0;

    tiny_solve_cpu(&solver);
}
