#include <sqp_solver/sqp_options.h>
#include <iostream>

SQPoptions::SQPoptions()
{
    defaults();
}

void SQPoptions::defaults()
{
    max_sqp_iters = 100;
    max_ls_iters = 20;
    min_ls_beta = 1e-4;
    ls_scale_factor = 0.5;
    tolerance = 1e-6;
}

void SQPoptions::print() const
{
    std::cout << "=== SQP Solver Options ===" << std::endl;
    std::cout << "  Max SQP iterations:    " << max_sqp_iters << std::endl;
    std::cout << "  Max linesearch iters:  " << max_ls_iters << std::endl;
    std::cout << "  Min linesearch beta:   " << min_ls_beta << std::endl;
    std::cout << "  Linesearch scale:      " << ls_scale_factor << std::endl;
    std::cout << "  Tolerance:             " << tolerance << std::endl;
}
