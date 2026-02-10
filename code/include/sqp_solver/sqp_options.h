struct SQPoptions {
    int max_sqp_iters;
    
    int max_ls_iters;
    double min_ls_beta;
    double ls_scale_factor;

    double tolerance;

    SQPoptions();
    void defaults();
    void print() const;
};
