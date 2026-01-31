#include <sqp_solver/sqp_solver.h>

#include <iostream>

SQPSolver::SQPSolver() { init(); }

SQPSolver::~SQPSolver() {
    // HPIPM memory automatically freed by HPIPMSolver destructor
}

void SQPSolver::solve() {
    for (int i = 0; i < _opts.max_sqp_iters; i++) {
        linearize();         // Compute Jacobians from dynamics/cost

        populate_qp();       // Transfer linearization to HPIPM (inlined)

        _qp_solver.solve();  // Direct HPIPM solve (inlined, warm-started)

        step();              // Update trajectory with QP solution

        if (break_criteria()) break;
    }
}

void SQPSolver::init() {
    // Pre-allocate ALL storage to avoid allocations in hot path
    // This is critical for performance!

    // Linearization storage
    _A.resize(_N * _nx * _nx);
    _B.resize(_N * _nx * _nu);
    _b.resize(_N * _nx);
    _Q.resize((_N + 1) * _nx * _nx);
    _R.resize(_N * _nu * _nu);
    _S.resize(_N * _nu * _nx);
    _q.resize((_N + 1) * _nx);
    _r.resize(_N * _nu);

    // Step storage (reused every iteration)
    _dx.resize(_nx);
    _du.resize(_nu);

    // Trajectory storage
    _x.resize((_N + 1) * _nx);
    _u.resize(_N * _nu);

    // Initialize HPIPM solver
    _qp_solver.initialize(_N, _nx, _nu, 0);  // ng=0 (no general constraints yet)

    // Set solver options for maximum performance
    _qp_solver.set_iter_max(50);             // Reasonable default
    _qp_solver.set_tol(1e-8, 1e-8, 1e-8, 1e-8);  // Tolerances
    _qp_solver.set_warm_start(true);         // CRITICAL: Enable warm-starting (huge speedup)
}

void SQPSolver::populate_qp() {
    // Transfer linearization data to HPIPM
    // These calls are INLINED by compiler → zero function call overhead

    // Dynamics for stages k=0..N-1
    for (int k = 0; k < _N; ++k) {
        _qp_solver.set_A(k, &_A[k * _nx * _nx]);
        _qp_solver.set_B(k, &_B[k * _nx * _nu]);
        _qp_solver.set_b(k, &_b[k * _nx]);
    }

    // Cost for stages k=0..N
    for (int k = 0; k <= _N; ++k) {
        _qp_solver.set_Q(k, &_Q[k * _nx * _nx]);
        _qp_solver.set_q(k, &_q[k * _nx]);

        if (k < _N) {
            _qp_solver.set_R(k, &_R[k * _nu * _nu]);
            _qp_solver.set_r(k, &_r[k * _nu]);
            // Note: S (cross term) can be added here if needed
            // _qp_solver.set_S(k, &_S[k * _nu * _nx]);
        }
    }
}

void SQPSolver::step() {
    // Extract QP solution and update trajectory
    // Cache-optimized with vectorizable loops

    double alpha = _ls_alpha;  // Line search step size

    // Use pre-allocated buffers (zero allocations!)
    for (int k = 0; k <= _N; ++k) {
        // Extract state step (inlined call)
        _qp_solver.get_x(k, _dx.data());

        // Update state (compiler can vectorize this with SIMD)
        double* x_k = &_x[k * _nx];
        for (int i = 0; i < _nx; ++i) {
            x_k[i] += alpha * _dx[i];
        }

        // Extract and update control (if not terminal stage)
        if (k < _N) {
            _qp_solver.get_u(k, _du.data());

            double* u_k = &_u[k * _nu];
            for (int i = 0; i < _nu; ++i) {
                u_k[i] += alpha * _du[i];
            }
        }
    }
}

void SQPSolver::linearize() {
    // TODO: Compute linearization of dynamics and cost
    // This will use Pinocchio to compute Jacobians
}

bool SQPSolver::break_criteria() {
    // TODO: Implement convergence check
    return false;
}
