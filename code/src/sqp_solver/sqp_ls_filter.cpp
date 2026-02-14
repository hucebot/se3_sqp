#include <sqp_solver/sqp_solver.h>

bool SQPSolver::ls_filter() {
     _candidate_cost   = _ocproblem.cost();
     _candidate_viol   = _ocproblem.constraint_violation();
     _candidate_defect = _ocproblem.dynamics_defect();

    if ( _candidate_cost   <  _nominal_cost   || std::fabs( _candidate_cost   -  _nominal_cost)   <= std::numeric_limits<double>::epsilon() ||
         _candidate_viol   <  _nominal_viol   || std::fabs( _candidate_viol   -  _nominal_viol)   <= std::numeric_limits<double>::epsilon() ||
         _candidate_defect <  _nominal_defect || std::fabs( _candidate_defect -  _nominal_defect) <= std::numeric_limits<double>::epsilon())
        return true;

    return false;
}
