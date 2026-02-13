enum class LSType { NONE, MERIT, FILTER };

struct SQPoptions {
    int max_sqp_iters;

    LSType ls_type;
    int max_ls_iters;
    double ls_scale_factor;

    double tolerance;

    SQPoptions();
    void defaults();
    void print() const;
};
