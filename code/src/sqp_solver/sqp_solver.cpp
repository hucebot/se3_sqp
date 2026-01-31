#include <sqp_solver/sqp_solver.h>

#include <cmath>
#include <iostream>

SQPSolver::SQPSolver() { init(); }

SQPSolver::~SQPSolver() {
    // HPIPM memory automatically freed by HPIPMSolver destructor
}

void SQPSolver::solve() {
    for (int i = 0; i < _opts.max_sqp_iters; i++) {
        linearize();  // Compute Jacobians from dynamics/cost

        populate_qp();  // Transfer linearization to HPIPM (inlined)

        _qp_solver.solve();  // Direct HPIPM solve (inlined, warm-started)

        step();  // Update trajectory with QP solution

        if (break_criteria()) break;
    }
}

void SQPSolver::init() {
    // Linearization storage - resize vectors and pre-allocate Eigen matrices
    _A.resize(_N);
    _B.resize(_N);
    _b.resize(_N);
    _Q.resize(_N + 1);
    _R.resize(_N);
    _S.resize(_N);
    _q.resize(_N + 1);
    _r.resize(_N);

    // Pre-allocate all matrices and vectors for zero allocations in hot path
    for (int k = 0; k < _N; ++k) {
        _A[k].resize(_ndx, _ndx);
        _B[k].resize(_ndx, _ndu);
        _b[k].resize(_ndx);
        _Q[k].resize(_ndx, _ndx);
        _R[k].resize(_ndu, _ndu);
        _S[k].resize(_ndu, _ndx);
        _q[k].resize(_ndx);
        _r[k].resize(_ndu);
    }
    // Terminal stage cost
    _Q[_N].resize(_ndx, _ndx);
    _q[_N].resize(_ndx);

    // Step storage (reused every iteration)
    _dx.resize(_N + 1);
    _du.resize(_N);
    for (int k = 0; k <= _N; ++k) {
        _dx[k].resize(_ndx);
        if (k < _N) {
            _du[k].resize(_ndu);
        }
    }

    // Trajectory storage
    _x.resize(_N + 1);
    _u.resize(_N);
    for (int k = 0; k <= _N; ++k) {
        _x[k].resize(_ndx);
        if (k < _N) {
            _u[k].resize(_ndu);
        }
    }

    // Initialize HPIPM solver
    _qp_solver.initialize(_N, _ndx, _ndu,
                          0);  // ng=0 (no general constraints yet)

    // Set solver options for maximum performance
    _qp_solver.set_iter_max(50);                 // Reasonable default
    _qp_solver.set_tol(1e-8, 1e-8, 1e-8, 1e-8);  // Tolerances
    _qp_solver.set_warm_start(
        true);  // CRITICAL: Enable warm-starting (huge speedup)
}

void SQPSolver::populate_qp() {
    // Transfer linearization data to HPIPM
    // Zero-copy: pass Eigen data pointers directly (column-major format)
    // HPIPM expects column-major (Fortran-style), which matches Eigen's default

    // Dynamics for stages k=0..N-1
    for (int k = 0; k < _N; ++k) {
        _qp_solver.set_A(k, _A[k].data());  // Zero-copy pointer to Eigen matrix
        _qp_solver.set_B(k, _B[k].data());
        _qp_solver.set_b(k, _b[k].data());  // Zero-copy pointer to Eigen vector
    }

    // Cost for stages k=0..N
    for (int k = 0; k <= _N; ++k) {
        _qp_solver.set_Q(k, _Q[k].data());
        _qp_solver.set_q(k, _q[k].data());

        if (k < _N) {
            _qp_solver.set_R(k, _R[k].data());
            _qp_solver.set_r(k, _r[k].data());
            // Cross term S can be added if needed:
            // _qp_solver.set_S(k, _S[k].data());
        }
    }
}

void SQPSolver::step() {
    // Extract QP solution and update trajectory
    // Eigen provides automatic SIMD vectorization (SSE/AVX)

    double alpha = _ls_alpha;  // Line search step size
    _step_norm = 0.0;          // Reset step norm for convergence check

    // Process all stages with zero-copy Eigen operations
    for (int k = 0; k <= _N; ++k) {
        // Extract state step from QP solution (zero-copy via Eigen::Map)
        _qp_solver.get_x(k, _dx[k].data());

        // Update state with vectorized operation (SIMD-optimized by Eigen)
        // Equivalent to: x_k += alpha * dx_k
        _x[k] += alpha * _dx[k];

        // Accumulate squared norm (Eigen optimizes this)
        _step_norm += (alpha * _dx[k]).squaredNorm();

        // Extract and update control (if not terminal stage)
        if (k < _N) {
            _qp_solver.get_u(k, _du[k].data());

            // Vectorized control update
            _u[k] += alpha * _du[k];

            // Accumulate squared norm
            _step_norm += (alpha * _du[k]).squaredNorm();
        }
    }

    // Take square root to get actual norm
    _step_norm = std::sqrt(_step_norm);
}

void SQPSolver::linearize() {
    // TODO: Compute linearization of dynamics and cost
    // Will use Pinocchio (which natively uses Eigen) to compute Jacobians
    // Pinocchio provides Eigen::MatrixXd outputs that can be directly assigned:
    //   _A[k] = pinocchio_jacobian_A;  // Direct Eigen matrix assignment
    //   _B[k] = pinocchio_jacobian_B;
    // This will be highly optimized with zero-copy when possible
}

bool SQPSolver::break_criteria() {
    // Check if the step norm is below tolerance (converged)
    // This is the 2-norm of the accepted step: ||alpha * [dx; du]||
    if (_step_norm < _opts.tolerance) {
        return true;  // Converged
    }

    return false;  // Not converged yet
}
