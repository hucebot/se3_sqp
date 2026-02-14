#include <sqp_solver/sqp_solver.h>

bool SQPSolver::ls_filter() {
    _last_cost   = _ocproblem.cost();
    _last_viol   = _ocproblem.constraint_violation();
    _last_defect = _ocproblem.dynamics_defect();

    if (_last_cost   < _prev_cost   || std::fabs(_last_cost   - _prev_cost)   <= std::numeric_limits<double>::epsilon() ||
        _last_viol   < _prev_viol   || std::fabs(_last_viol   - _prev_viol)   <= std::numeric_limits<double>::epsilon() ||
        _last_defect < _prev_defect || std::fabs(_last_defect - _prev_defect) <= std::numeric_limits<double>::epsilon())
        return true;

    return false;
}
