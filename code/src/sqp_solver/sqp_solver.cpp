#include <sqp_solver/sqp_solver.h>

#include <iostream>

SQPSolver::SQPSolver() { init(); }

SQPSolver::~SQPSolver() {
    // #TODO - Free memmory here
}

void SQPSolver::solve() {
    linearize();

    for (int i = 0; i < _opts.max_sqp_iters; i++) {
        _solution;  // #TODO - HPIPM_SOLVE_QP

        step();

        if (break_criteria()) break;
        // else
        // _stats.update_iter(); // #TODO - add this in stats
    }
}

void SQPSolver::init() {}
void SQPSolver::linearize() {}
void SQPSolver::step() {}
bool SQPSolver::break_criteria() {}
