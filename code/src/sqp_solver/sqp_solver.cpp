#include <sqp_solver/sqp_solver.h>

#include <cmath>

SQPSolver::SQPSolver(OCP& ocp): _ocproblem(ocp) { init(); }

SQPSolver::~SQPSolver() {
    // HPIPM memory automatically freed by HPIPMSolver destructor
}

void SQPSolver::solve() {
    _stats.start_timer();
    for (int i = 0; i < _opts.max_sqp_iters; i++) {
        _ls_alpha = 1.;
        linearize();
        populate_qp();
        _qp_solver.solve();
        step();

        _stats.update();
        _stats.print();
        if (break_criteria()) break;

    }
    _stats.print(1);
}

void SQPSolver::init() {
    _N = _ocproblem.num_nodes();
    _nx = _ocproblem.get_node(0).nx();
    _nu = _ocproblem.get_node(0).nu();
    _ndx = _nx;  // For now, assume delta-state = state dimension
    _ndu = _nu;
    _ls_alpha = 1.0;  // Default step size

    // Linearization storage - resize vectors and pre-allocate Eigen matrices
    // Dynamics: N-1 transitions (stages 0 to N-2)
    _A.resize(_N - 1);
    _B.resize(_N - 1);
    _b.resize(_N - 1);
    // Costs: N stages (0 to N-1)
    _Q.resize(_N);
    _R.resize(_N);
    _S.resize(_N);
    _q.resize(_N);
    _r.resize(_N);

    // Pre-allocate and zero dynamics matrices (N-1 transitions)
    for (int k = 0; k < _N - 1; ++k) {
        _A[k].setZero(_ndx, _ndx);
        _B[k].setZero(_ndx, _ndu);
        _b[k].setZero(_ndx);
    }
    // Pre-allocate and zero cost matrices (N stages)
    for (int k = 0; k < _N; ++k) {
        _Q[k].setIdentity(_ndx, _ndx);  // Default: identity cost on state
        _R[k].setIdentity(_ndu, _ndu);  // Default: identity cost on control
        _S[k].setZero(_ndu, _ndx);
        _q[k].setZero(_ndx);
        _r[k].setZero(_ndu);
    }

    // Step storage (reused every iteration)
    _dx.resize(_N);
    _du.resize(_N);
    for (int k = 0; k < _N; ++k) {
        _dx[k].setZero(_ndx);
        _du[k].setZero(_ndu);
    }

    // Trajectory storage
    _x_candidate.resize(_N);
    _u_candidate.resize(_N);
    for (int k = 0; k < _N; ++k) {
        _x_candidate[k].setZero(_ndx);
        _u_candidate[k].setZero(_ndu);
    }

    // Initialize HPIPM solver
    // HPIPM horizon = N-1 for N nodes (N-1 dynamics, terminal stage has no control)
    _qp_solver.initialize(_N - 1, _ndx, _ndu,
                          0);  // ng=0 (no general constraints yet)

    // Set solver options for maximum performance
    _qp_solver.set_iter_max(50);                 // Reasonable default
    _qp_solver.set_tol(1e-3, 1e-3, 1e-3, 1e-3);  // Tolerances
    _qp_solver.set_warm_start(
        true);  // CRITICAL: Enable warm-starting (huge speedup)
}

void SQPSolver::populate_qp() {
    // Transfer linearization data to HPIPM
    // Zero-copy: pass Eigen data pointers directly (column-major format)
    // HPIPM expects column-major (Fortran-style), which matches Eigen's default

    // Dynamics for stages k=0..N-2
    for (int k = 0; k < _N - 1; ++k) {
        _qp_solver.set_A(k, _A[k].data());
        _qp_solver.set_B(k, _B[k].data());
        _qp_solver.set_b(k, _b[k].data());
    }

    // Cost for HPIPM stages 0..N-1 (N stages total with horizon N-1)
    for (int k = 0; k < _N; ++k) {
        _qp_solver.set_Q(k, _Q[k].data());
        _qp_solver.set_q(k, _q[k].data());
        // Terminal stage (k = N-1) has no control
        if (k < _N - 1) {
            _qp_solver.set_R(k, _R[k].data());
            _qp_solver.set_r(k, _r[k].data());
        }
    }
}

void SQPSolver::step() {
    // Extract QP solution and compute candidate trajectory

    _step_norm = 0.0;          // Reset step norm for convergence check

    auto& x_nom = _ocproblem.x_traj();
    auto& u_nom = _ocproblem.u_traj();

    // Process all N stages (HPIPM stages 0 to N-1)
    for (int k = 0; k < _N; ++k) {
        _qp_solver.get_x(k, _dx[k].data());
        _step_norm += (_ls_alpha * _dx[k]).squaredNorm();

        _x_candidate[k] = x_nom[k] + _ls_alpha * _dx[k]; //TODO - This needs the \oplus

        // Terminal stage (k = N-1) has no control
        if (k < _N - 1) {
            _qp_solver.get_u(k, _du[k].data());
            _step_norm += (_ls_alpha * _du[k]).squaredNorm();

            _u_candidate[k] = u_nom[k] + _ls_alpha * _du[k];
        }
    }

    // Accept step: copy candidate to nominal
    // TODO: This will be moved to line search logic
    for (int k = 0; k < _N; ++k) {
        x_nom[k] = _x_candidate[k];
        if (k < _N - 1) {
            u_nom[k] = _u_candidate[k];
        }
    }

    // Take square root to get actual norm
    _step_norm = std::sqrt(_step_norm);
    _stats.update_step_norm(_step_norm);
}

void SQPSolver::linearize() {
    for (int i = 0; i < _N; i++)
    {
        // Dynamics for N-1 transitions (nodes 0 to N-2)
        if (i < _N - 1)
        {
            auto dyn = _ocproblem.get_node(i).get_dynamics();
            dyn->evaluate(_b[i]);
            dyn->jacobian();
            _A[i] = dyn->get_jac_x();
            _B[i] = dyn->get_jac_u();
        }

        // TODO: Cost linearization for all N nodes
    }
}

bool SQPSolver::break_criteria() {
    // Check if the step norm is below tolerance (converged)
    // This is the 2-norm of the accepted step: ||alpha * [dx; du]||
    if (_step_norm < _opts.tolerance) {
        return true;  // Converged
    }

    return false;  // Not converged yet
}
