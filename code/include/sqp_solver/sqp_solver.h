#pragma once

#include <sqp_solver/sqp_options.h>
#include <sqp_solver/sqp_statistics.h>

extern "C" {
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_dim.h>
}

class SQPSolver {
   private:
    int _N, _nx, _nu;
    int _solution;  // #FIXME - not an int fix it
    double _ls_alpha;

    SQPstatistics _stats;
    SQPoptions _opts;

    void init();
    void linearize();
    void step();
    bool* linesearch();
    bool ls_merit();
    bool ls_filter();
    bool break_criteria();

   public:
    SQPSolver();
    ~SQPSolver();

    /// @brief solve the defined problem
    void solve();
};