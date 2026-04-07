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
    std::vector<VectorXd> _dx;         // State steps: _dx[k] is VectorXd of size _ndx
    std::vector<VectorXd> _du;         // Control steps: _du[k] is VectorXd of size _ndu
    std::vector<VectorXd> _scaled_dx;  // Scratch: alpha * _dx[k], reused across step() calls
    std::vector<VectorXd> _scaled_du;  // Scratch: alpha * _du[k], reused across step() calls

    // Lagrange multipliers (dual variables from QP)
    std::vector<VectorXd> _pi;         // Dynamics multipliers: _pi[k], k=0..N-2, each ndx
    std::vector<VectorXd> _lam_lg;     // General constraint lower multipliers, each ng
    std::vector<VectorXd> _lam_ug;     // General constraint upper multipliers, each ng

    // QP-solution multipliers (raw from QP solve)
    std::vector<VectorXd> _pi_qp;
    std::vector<VectorXd> _lam_lg_qp;
    std::vector<VectorXd> _lam_ug_qp;

    // Candidate multipliers (blended with α, analogous to _x_candidate/_u_candidate)
    std::vector<VectorXd> _pi_candidate;
    std::vector<VectorXd> _lam_lg_candidate;
    std::vector<VectorXd> _lam_ug_candidate;

    double _ls_alpha;
    double _step_norm;  // Norm of the accepted step (for convergence check)
    double _current_reg; // Current adaptive regularization value


    double  _nominal_cost;
    double  _nominal_defect;
    double  _nominal_viol;

    // Values at the last accepted candidate (populated by ls_filter / LSType::NONE path)
    double  _candidate_cost   = 0.;
    double  _candidate_viol   = 0.;
    double  _candidate_defect = 0.;

    // Initial state constraint: dx_0 = 0 (fix first state perturbation)
    std::vector<int> _idxbx;  // Indices of constrained state variables at stage 0
    VectorXd _lbx;            // Lower bound (zeros)
    VectorXd _ubx;            // Upper bound (zeros)

    SQPstatistics _stats;
    SQPoptions _opts;

    /// Allocate all storage and configure QP solver from OCP dimensions
    void init();

    /// Rebind nodes to nominal trajectory, then linearize dynamics, costs,
    /// and constraints around the current nominal. Populates A,B,b,Q,q,R,r,C,D,lg,ug.
    void linearize();

    /// Transfer linearization matrices into the HPIPM QP solver
    void populate_qp();

    /// Compute candidate trajectory from QP step (_dx,_du) at current _ls_alpha.
    /// Binds nodes to _x_candidate/_u_candidate so ls functions can evaluate directly.
    void step();

    /// Swap candidate trajectory into nominal (O(1)) and rebind nodes
    void accept_step();

    /// Function pointer to active line search (set once in init from _opts.ls_type).
    /// nullptr when LSType::NONE.
    bool (SQPSolver::*_ls_function)();

    /// L1 merit function check. Nodes are already bound to candidates by step().
    /// Returns true if Armijo condition is satisfied.
    bool ls_merit();

    /// Filter-based acceptance check. Nodes are already bound to candidates by step().
    /// Returns true if candidate is acceptable to the filter.
    bool ls_filter();

    /// Compute NLP-level KKT stationarity residual using multipliers
    double compute_kkt_residual();

    /// Check convergence (KKT + feasibility, with step norm fallback)
    bool break_criteria();

   public:
    SQPSolver(OCP& ocp);
    ~SQPSolver();

    /// @brief set solver options before calling solve()
    void set_options(const SQPoptions& opts);

    /// @brief solve the defined problem
    void solve(const Eigen::VectorXd& x0 = Eigen::VectorXd(0));

    /// @brief access solver statistics after calling solve()
    const SQPstatistics& get_stats() const { return _stats; }
};