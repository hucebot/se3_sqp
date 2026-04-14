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
    ls_merit_mu  = 1.0;
    tolerance = 1e-3;
    regularization = 1e-12;
    regularization_scale = 1e3;
    eps_inequality = 1e-4;
    verbose = 1;
    hpipm_iter_max   = 1000;
    hpipm_tol_stat   = 1e-4;
    hpipm_tol_eq     = 1e-8;
    hpipm_tol_ineq   = 1e-8;
    hpipm_tol_comp   = 1e-4;
    hpipm_warm_start = false;
}

void SQPoptions::print() const
{
    if (verbose == 0) return;
    const char* ls_name = ls_type == LSType::MERIT  ? "MERIT"  :
                          ls_type == LSType::FILTER ? "FILTER" : "NONE";
    std::cout << "=== SQP Solver Options ===" << std::endl;
    std::cout << "  Max SQP iterations:    " << max_sqp_iters << std::endl;
    std::cout << "  Tolerance:             " << tolerance << std::endl;
    std::cout << "  Regularization:        " << regularization << std::endl;
    std::cout << "  Regularization scale:  " << regularization_scale << std::endl;
    std::cout << "  Eps inequality:             " << eps_inequality << std::endl;
    std::cout << "  Verbose:               " << verbose << std::endl;
    std::cout << "  Linesearch type:       " << ls_name << std::endl;
    if (ls_type != LSType::NONE) {
        std::cout << "  Max linesearch iters:  " << max_ls_iters << std::endl;
        std::cout << "  Linesearch scale:      " << ls_scale_factor << std::endl;
        if (ls_type == LSType::MERIT) {
            std::cout << "  Armijo decrease (eta): " << ls_merit_eta << std::endl;
            std::cout << "  L1 penalty (mu):       " << ls_merit_mu << std::endl;
        }
    }
    std::cout << "  HPIPM iter max:        " << hpipm_iter_max << std::endl;
    std::cout << "  HPIPM tol (stat/eq/ineq/comp): "
              << hpipm_tol_stat << " / " << hpipm_tol_eq << " / "
              << hpipm_tol_ineq << " / " << hpipm_tol_comp << std::endl;
    std::cout << "  HPIPM warm start:      " << (hpipm_warm_start ? "yes" : "no") << std::endl;
}
