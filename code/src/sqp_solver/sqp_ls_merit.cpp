#include <sqp_solver/sqp_solver.h>

#include <algorithm>
#include <cmath>
#include <iostream>

bool SQPSolver::ls_merit() {

    // Candidate values (nodes already bound to x_candidate by step())
    _candidate_cost   = _ocproblem.cost();
    _candidate_defect = _ocproblem.dynamics_defect();
    _candidate_viol   = _ocproblem.constraint_violation();

    const double phi_alpha = _candidate_cost + _opts.ls_merit_mu * (_candidate_defect + _candidate_viol);
    const double phi_0     = _nominal_cost   + _opts.ls_merit_mu * (_nominal_defect   + _nominal_viol);

    // Directional derivative φ'(0) = ∇φ · (dx, du)
    // Gradients were computed at the nominal in linearize() via calc_*_gradient().
    double dphi = 0.0;
    for (int k = 0; k < _N; ++k) {
        const Node& node = _ocproblem.get_node(k);

        dphi += node.get_cost_grad_x().dot(_dx[k]);
        dphi += _opts.ls_merit_mu * node.get_defect_grad_x().dot(_dx[k]);
        dphi += _opts.ls_merit_mu * node.get_violation_grad_x().dot(_dx[k]);

        if (k < _Nu) {
            dphi += node.get_cost_grad_u().dot(_du[k]);
            dphi += _opts.ls_merit_mu * node.get_defect_grad_u().dot(_du[k]);
            dphi += _opts.ls_merit_mu * node.get_violation_grad_u().dot(_du[k]);
        }
    }

    return phi_alpha <= phi_0 + _opts.ls_merit_eta * _ls_alpha * dphi;
}
