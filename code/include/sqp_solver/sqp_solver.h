#pragma once

#include <common/types.h>
#include <qp_solver/hpipm_solver.h>
#include <sqp_solver/sqp_options.h>
#include <sqp_solver/sqp_statistics.h>
#include <trajopt/ocp.h>

#include <Eigen/Dense>
#include <vector>

class SQPSolver {
   private:
    int _N, _Nx, _Nu, _nx, _ndx, _nu,
        _ndu;  // number of nodes, state, deltastate, control, deltacontrol

    OCP& _ocproblem;


    // High-performance QP solver (direct member for zero indirection)
    HPIPMSolver _qp_solver;

    // // Current trajectory (Eigen for SIMD-vectorized operations)
    std::vector<VectorXd> _x_candidate;  // States: _x[k] is VectorXd of size _nx
    std::vector<VectorXd> _u_candidate;  // Controls: _u[k] is VectorXd of size _nu
    
    // Linearization storage per stage (Eigen matrices for optimized operations)
    std::vector<MatrixXd> _A;  // Dynamics Jacobian A_k (nx x nx)
    std::vector<MatrixXd> _B;  // Dynamics Jacobian B_k (nx x nu)
    std::vector<VectorXd> _b;  // Dynamics affine term (nx)
    std::vector<MatrixXd> _Q;  // State cost Hessian (nx x nx)
    std::vector<MatrixXd> _R;  // Control cost Hessian (nu x nu)
    std::vector<MatrixXd> _S;  // Cross term (nu x nx)
    std::vector<VectorXd> _q;  // Linear state cost (nx)
    std::vector<VectorXd> _r;  // Linear control cost (nu)

    // General constraints: lg_k <= C_k*dx + D_k*du <= ug_k
    int _ng;                   // Number of general constraints per stage (uniform)
    std::vector<MatrixXd> _C;  // Constraint Jacobian w.r.t. state (ng x nx)
    std::vector<MatrixXd> _D;  // Constraint Jacobian w.r.t. control (ng x nu)
    std::vector<VectorXd> _lg; // Lower bound: lb - g(x) (ng)
    std::vector<VectorXd> _ug; // Upper bound: ub - g(x) (ng)

    // Step storage for full trajectory (pre-allocated for zero allocations)
    std::vector<VectorXd> _dx;  // State steps: _dx[k] is VectorXd of size _ndx
    std::vector<VectorXd>
        _du;  // Control steps: _du[k] is VectorXd of size _ndu

    double _ls_alpha;
    double _step_norm;  // Norm of the accepted step (for convergence check)

    // Initial state constraint: dx_0 = 0 (fix first state perturbation)
    std::vector<int> _idxbx;  // Indices of constrained state variables at stage 0
    VectorXd _lbx;            // Lower bound (zeros)
    VectorXd _ubx;            // Upper bound (zeros)

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
    SQPSolver(OCP& ocp);
    ~SQPSolver();

    /// @brief solve the defined problem
    void solve();
};