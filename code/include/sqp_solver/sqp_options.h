enum class LSType { NONE, MERIT, FILTER };

struct SQPoptions {
    int max_sqp_iters;

    LSType ls_type;
    int max_ls_iters;
    double ls_scale_factor;

    double tolerance;
    double regularization;        // Hessian regularization: Q += reg*I, R += reg*I
    double regularization_scale;  // Scale-up factor on QP failure (adaptive)

    // TODO add verbose level option

    SQPoptions();
    void defaults();
    void print() const;
};
