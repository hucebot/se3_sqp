#include <sqp_solver/sqp_options.h>
#include <iostream>

SQPoptions::SQPoptions()
{
    defaults();
}

void SQPoptions::defaults()
{
    max_sqp_iters = 100;
    ls_type = LSType::NONE;
    max_ls_iters = 5;
    ls_scale_factor = 0.5;
    ls_merit_eta = 1e-4;
    tolerance = 1e-3;
    regularization = 1e-12;
    regularization_scale = 1e3;
}

void SQPoptions::print() const
{
    std::cout << "=== SQP Solver Options ===" << std::endl;
    auto ls_name = ls_type == LSType::MERIT ? "MERIT" :
                   ls_type == LSType::FILTER ? "FILTER" : "NONE";
    std::cout << "  Max SQP iterations:    " << max_sqp_iters << std::endl;
    std::cout << "  Linesearch type:       " << ls_name << std::endl;
    std::cout << "  Max linesearch iters:  " << max_ls_iters << std::endl;
    std::cout << "  Linesearch scale:      " << ls_scale_factor << std::endl;
    std::cout << "  Armijo decrease:       " << ls_merit_eta << std::endl;
    std::cout << "  Tolerance:             " << tolerance << std::endl;
    std::cout << "  Regularization:        " << regularization << std::endl;
    std::cout << "  Regularization scale:  " << regularization_scale << std::endl;
}
