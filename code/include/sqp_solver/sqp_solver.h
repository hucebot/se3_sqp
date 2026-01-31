#pragma once

#include <qp_solver/hpipm_solver.h>
#include <sqp_solver/sqp_options.h>
#include <sqp_solver/sqp_statistics.h>

#include <Eigen/Dense>
#include <vector>

class SQPSolver {
   private:
    int _N, _nx, _ndx, _nu,
        _ndu;  // number of nodes, state, deltastate, control, deltacontrol

    // High-performance QP solver (direct member for zero indirection)
    HPIPMSolver _qp_solver;

    // Current trajectory (Eigen for SIMD-vectorized operations)
    std::vector<Eigen::VectorXd> _x;  // States: _x[k] is VectorXd of size _nx
    std::vector<Eigen::VectorXd> _u;  // Controls: _u[k] is VectorXd of size _nu

    // Linearization storage per stage (Eigen matrices for optimized operations)
    std::vector<Eigen::MatrixXd> _A;  // Dynamics Jacobian A_k (nx x nx)
    std::vector<Eigen::MatrixXd> _B;  // Dynamics Jacobian B_k (nx x nu)
    std::vector<Eigen::VectorXd> _b;  // Dynamics affine term (nx)
    std::vector<Eigen::MatrixXd> _Q;  // State cost Hessian (nx x nx)
    std::vector<Eigen::MatrixXd> _R;  // Control cost Hessian (nu x nu)
    std::vector<Eigen::MatrixXd> _S;  // Cross term (nu x nx)
    std::vector<Eigen::VectorXd> _q;  // Linear state cost (nx)
    std::vector<Eigen::VectorXd> _r;  // Linear control cost (nu)

    // Step storage for full trajectory (pre-allocated for zero allocations)
    std::vector<Eigen::VectorXd>
        _dx;  // State steps: _dx[k] is VectorXd of size _ndx
    std::vector<Eigen::VectorXd>
        _du;  // Control steps: _du[k] is VectorXd of size _ndu

    double _ls_alpha;
    double _step_norm;  // Norm of the accepted step (for convergence check)

    SQPstatistics _stats;
    SQPoptions _opts;

    void init();
    void linearize();
    void populate_qp();  // Transfer linearization to QP solver
    void step();
    bool* linesearch();
    bool ls_merit();
    bool ls_filter();
    bool break_criteria();

   public:
    SQPSolver();
    ~SQPSolver();

    /// @brief solve the defined problem
    void solve();
};