#pragma once

enum class LSType { NONE, MERIT, FILTER };

struct SQPoptions {
    int max_sqp_iters;

    LSType ls_type;
    int max_ls_iters;
    double ls_scale_factor;
    double ls_merit_eta;   // Armijo sufficient decrease parameter
    double ls_merit_mu;    // L1 penalty weight for merit function

    double tolerance;
    double regularization;        // Hessian regularization: Q += reg*I, R += reg*I
    double regularization_scale;  // Scale-up factor on QP failure (adaptive)

    int verbose;  // 0=silent, 1=per-iteration + final summary, 2=final only (MPC)

    // HPIPM inner QP solver settings
    int    hpipm_iter_max;
    double hpipm_tol_stat;
    double hpipm_tol_eq;
    double hpipm_tol_ineq;
    double hpipm_tol_comp;
    bool   hpipm_warm_start;

    SQPoptions();
    void defaults();
    void print() const;
};
