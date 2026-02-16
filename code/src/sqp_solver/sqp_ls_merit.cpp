#include <sqp_solver/sqp_solver.h>

#include <algorithm>
#include <cmath>
#include <iostream>

bool SQPSolver::ls_merit() {
    // L1 merit function: φ(α) = cost + μ*(defect + violation)
    // Armijo condition:  φ(α) ≤ φ(0) + η·α·φ'(0)
    //
    // TODO: promote mu / eta to SQPoptions when tuning needs arise.
    constexpr double mu  = 1.0;   // L1 penalty weight
    constexpr double eta = 1e-4;   // Armijo sufficient-decrease constant

    // Candidate values (nodes already bound to x_candidate by step())
    _candidate_cost   = _ocproblem.cost();
    _candidate_defect = _ocproblem.dynamics_defect();
    _candidate_viol   = _ocproblem.constraint_violation();

    const double phi_alpha =  _candidate_cost + mu * ( _candidate_defect +  _candidate_viol);
    const double phi_0     =  _nominal_cost + mu * ( _nominal_defect +  _nominal_viol);

    // Directional derivative φ'(0) = ∇φ · (dx, du)
    // Gradients were computed at the nominal in linearize() via calc_*_gradient().
    double dphi = 0.0;
    for (int k = 0; k < _N; ++k) {
        const Node& node = _ocproblem.get_node(k);

        dphi += node.get_cost_grad_x().dot(_dx[k]);
        dphi += mu * node.get_defect_grad_x().dot(_dx[k]);
        dphi += mu * node.get_violation_grad_x().dot(_dx[k]);

        if (k < _Nu) {
            dphi += node.get_cost_grad_u().dot(_du[k]);
            dphi += mu * node.get_defect_grad_u().dot(_du[k]);
            dphi += mu * node.get_violation_grad_u().dot(_du[k]);
        }
    }

    // If dphi >= 0 the merit can't decrease (no cost, or already feasible):
    // accept immediately to avoid stalling on a degenerate Armijo condition.
    if (dphi >= 0.0) return true;

    return phi_alpha <= phi_0 + eta * _ls_alpha * dphi;
}
