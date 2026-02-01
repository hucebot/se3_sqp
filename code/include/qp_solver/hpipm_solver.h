#pragma once

#include <Eigen/Dense>

extern "C" {
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp_ipm.h>
#include <hpipm_d_ocp_qp_sol.h>
}

/**
 * @brief High-performance header-only wrapper for HPIPM OCP QP solver
 *
 * All methods are inline for maximum performance (zero function call overhead).
 * Compiler can inline across translation units with LTO enabled.
 */
class HPIPMSolver {
   private:
    int _N, _nx, _nu, _ng;
    bool _initialized;

    // HPIPM structures
    d_ocp_qp_dim _qp_dim;
    d_ocp_qp _qp;
    d_ocp_qp_sol _qp_sol;
    d_ocp_qp_ipm_arg _qp_arg;
    d_ocp_qp_ipm_ws _qp_ws;

    // Memory arenas (HPIPM uses custom allocators)
    void* _dim_mem;
    void* _qp_mem;
    void* _sol_mem;
    void* _arg_mem;
    void* _ws_mem;

   public:
    HPIPMSolver();
    ~HPIPMSolver();

    // Disable copy/move (HPIPM memory is non-trivial)
    HPIPMSolver(const HPIPMSolver&) = delete;
    HPIPMSolver& operator=(const HPIPMSolver&) = delete;

    // Initialization (only non-inline method besides destructor)
    void initialize(int N, int nx, int nu, int ng = 0, int nbx = 0, int nbu = 0);

    // === Problem Specification (ALL INLINE for zero overhead) ===

    /** Set dynamics matrix A_k (nx x nx) for stage k */
    inline void set_A(int stage, double* A) { d_ocp_qp_set_A(stage, A, &_qp); }

    /** Set dynamics matrix B_k (nx x nu) for stage k */
    inline void set_B(int stage, double* B) { d_ocp_qp_set_B(stage, B, &_qp); }

    /** Set dynamics affine term b_k (nx x 1) for stage k */
    inline void set_b(int stage, double* b) { d_ocp_qp_set_b(stage, b, &_qp); }

    /** Set state cost matrix Q_k (nx x nx) for stage k */
    inline void set_Q(int stage, double* Q) { d_ocp_qp_set_Q(stage, Q, &_qp); }

    /** Set control cost matrix R_k (nu x nu) for stage k */
    inline void set_R(int stage, double* R) { d_ocp_qp_set_R(stage, R, &_qp); }

    /** Set cross term S_k (nu x nx) for stage k */
    inline void set_S(int stage, double* S) { d_ocp_qp_set_S(stage, S, &_qp); }

    /** Set linear state cost q_k (nx x 1) for stage k */
    inline void set_q(int stage, double* q) { d_ocp_qp_set_q(stage, q, &_qp); }

    /** Set linear control cost r_k (nu x 1) for stage k */
    inline void set_r(int stage, double* r) { d_ocp_qp_set_r(stage, r, &_qp); }

    // === State and Control Bounds ===

    /** Set lower bounds on state x_k for stage k */
    inline void set_lbx(int stage, double* lbx) {
        d_ocp_qp_set_lbx(stage, lbx, &_qp);
    }

    /** Set upper bounds on state x_k for stage k */
    inline void set_ubx(int stage, double* ubx) {
        d_ocp_qp_set_ubx(stage, ubx, &_qp);
    }

    /** Set lower bounds on control u_k for stage k */
    inline void set_lbu(int stage, double* lbu) {
        d_ocp_qp_set_lbu(stage, lbu, &_qp);
    }

    /** Set upper bounds on control u_k for stage k */
    inline void set_ubu(int stage, double* ubu) {
        d_ocp_qp_set_ubu(stage, ubu, &_qp);
    }

    /** Set state indices for box constraints at stage k */
    inline void set_idxbx(int stage, int* idxbx) {
        d_ocp_qp_set_idxbx(stage, idxbx, &_qp);
    }

    /** Set control indices for box constraints at stage k */
    inline void set_idxbu(int stage, int* idxbu) {
        d_ocp_qp_set_idxbu(stage, idxbu, &_qp);
    }

    // === Solve (INLINE - Direct HPIPM call) ===

    /**
     * Solve the QP problem
     * @note This is warm-started if set_warm_start(true) was called
     * HPIPM returns void, use get_status() to check convergence
     */
    inline void solve() {
        d_ocp_qp_ipm_solve(&_qp, &_qp_sol, &_qp_arg, &_qp_ws);
    }

    // === Solution Extraction (ALL INLINE) ===
    // Note: Can't be const because HPIPM doesn't use const (poor API design)

    /** Extract control solution u_k for stage k */
    inline void get_u(int stage, double* u_out) {
        d_ocp_qp_sol_get_u(stage, &_qp_sol, u_out);
    }

    /** Extract state solution x_k for stage k */
    inline void get_x(int stage, double* x_out) {
        d_ocp_qp_sol_get_x(stage, &_qp_sol, x_out);
    }

    /** Extract dynamics multipliers pi_k for stage k */
    inline void get_pi(int stage, double* pi_out) {
        d_ocp_qp_sol_get_pi(stage, &_qp_sol, pi_out);
    }

    // === Solver Options (ALL INLINE) ===

    /** Set maximum IPM iterations */
    inline void set_iter_max(int max_iter) {
        d_ocp_qp_ipm_arg_set_iter_max(&max_iter, &_qp_arg);
    }

    /** Set convergence tolerances */
    inline void set_tol(double tol_stat, double tol_eq, double tol_ineq,
                        double tol_comp) {
        d_ocp_qp_ipm_arg_set_tol_stat(&tol_stat, &_qp_arg);
        d_ocp_qp_ipm_arg_set_tol_eq(&tol_eq, &_qp_arg);
        d_ocp_qp_ipm_arg_set_tol_ineq(&tol_ineq, &_qp_arg);
        d_ocp_qp_ipm_arg_set_tol_comp(&tol_comp, &_qp_arg);
    }

    /**
     * Enable/disable warm-starting
     * @param enable If true, reuses previous solution (HUGE performance gain)
     */
    inline void set_warm_start(bool enable) {
        int warm_start = enable ? 1 : 0;
        d_ocp_qp_ipm_arg_set_warm_start(&warm_start, &_qp_arg);
    }

    // === Direct access (if needed for advanced usage) ===

    d_ocp_qp* get_qp() { return &_qp; }
    d_ocp_qp_sol* get_sol() { return &_qp_sol; }
};
