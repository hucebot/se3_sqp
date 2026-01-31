#pragma once

#include <sqp_solver/sqp_options.h>
#include <sqp_solver/sqp_statistics.h>
#include <qp_solver/hpipm_solver.h>

#include <vector>

class SQPSolver {
   private:
    int _N, _nx, _nu;

    // High-performance QP solver (direct member for zero indirection)
    HPIPMSolver _qp_solver;

    // Current trajectory
    std::vector<double> _x;  // States: _x[k*_nx + i]
    std::vector<double> _u;  // Controls: _u[k*_nu + i]

    // Linearization storage (pre-allocated for zero allocations in hot path)
    std::vector<double> _A;  // Dynamics Jacobian A_k
    std::vector<double> _B;  // Dynamics Jacobian B_k
    std::vector<double> _b;  // Dynamics affine term
    std::vector<double> _Q;  // State cost Hessian
    std::vector<double> _R;  // Control cost Hessian
    std::vector<double> _S;  // Cross term
    std::vector<double> _q;  // Linear state cost
    std::vector<double> _r;  // Linear control cost

    // Step storage (pre-allocated)
    std::vector<double> _dx;
    std::vector<double> _du;

    double _ls_alpha;

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