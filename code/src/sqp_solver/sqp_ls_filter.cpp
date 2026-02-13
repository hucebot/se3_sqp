#include <sqp_solver/sqp_solver.h>

bool SQPSolver::ls_filter() {
    // TODO optimize all these calls
    DEBUG_PRINT("FILTER_LS");
    if (_ocproblem.cost() <  _prev_cost || std::fabs(_ocproblem.cost() -  _prev_cost) <=  std::numeric_limits<double>::epsilon() ||
        _ocproblem.constraint_violation() < _prev_viol || std::fabs(_ocproblem.constraint_violation() - _prev_viol) <=  std::numeric_limits<double>::epsilon() ||
        _ocproblem.dynamics_defect() < _prev_defect || std::fabs(_ocproblem.dynamics_defect() - _prev_defect)  <=  std::numeric_limits<double>::epsilon())
        return true;

    return false;
}
