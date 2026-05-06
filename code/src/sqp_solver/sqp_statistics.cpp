#include <sqp_solver/sqp_statistics.h>
#include <iostream>
#include <iomanip>

SQPstatistics::SQPstatistics()
{
    reset();
}

void SQPstatistics::reset()
{
    number_of_iterations = 0;
    total_cost = 0.0;
    total_constraint_violation = 0.0;
    total_dynamics_defect = 0.0;
    step_norm = 0.0;
    dual_infeasibility = 0.0;
    linesearch_iterations = 0;
    qp_status = 0;
    qp_iterations = 0;
    total_time_ms = 0.0;
    last_iteration_time_ms = 0.0;
}

void SQPstatistics::print(int verbosity) const
{
    print_internal(std::cout, verbosity);
}

void SQPstatistics::print_to_file(const std::string& filename, int verbosity) const
{
    std::ofstream file(filename, std::ios::app);
    if (file.is_open())
    {
        print_internal(file, verbosity);
        file.close();
    }
    else
    {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
    }
}

void SQPstatistics::print_internal(std::ostream& os, int verbosity) const
{
    os << std::scientific << std::setprecision(2);

    if (verbosity == 0)
    {
        // Normal output - header at beginning and every 10 iterations
        if (number_of_iterations == 1 || number_of_iterations % 10 == 0)
        {
            os << std::setw(6) << "Iter"
               << std::setw(10) << "Cost"
               << std::setw(10) << "Violation"
               << std::setw(10) << "Defect"
               << std::setw(10) << "DualInf"
               << std::setw(10) << "||d||"
               << std::setw(4) << "LS"
               << std::setw(10) << "QP[s/i]"
               << std::setw(8) << "t(ms)" << std::endl;
        }
        os << std::setw(6) << number_of_iterations
           << std::setw(10) << total_cost
           << std::setw(10) << total_constraint_violation
           << std::setw(10) << total_dynamics_defect
           << std::setw(10) << dual_infeasibility
           << std::setw(10) << step_norm
           << std::setw(4) << linesearch_iterations
           << std::setw(4) << qp_status << " /" << std::setw(4) << qp_iterations
           << std::fixed << std::setw(8) << std::setprecision(3) << last_iteration_time_ms << std::endl;
    }
    else if (verbosity == 1)
    {
        // MPC mode: one summary line per solve() call
        static int mpc_step = 0;
        if (mpc_step % 10 == 0)
        {
            os << std::setw(5) << "SQPi"
               << std::setw(10) << "t(ms)"
               << std::setw(10) << "Cost"
               << std::setw(10) << "Violation"
               << std::setw(10) << "Defect"
               << std::setw(10) << "DualInf" << std::endl;
        }
        os << std::fixed << std::setprecision(3)
           << std::setw(5)  << number_of_iterations
           << std::setw(10) << total_time_ms 
           << std::setw(10) << total_cost
           << std::setw(10) << total_constraint_violation
           << std::setw(10) << total_dynamics_defect
           << std::setw(10) << dual_infeasibility << std::endl;
        ++mpc_step;
    }
    else if (verbosity == 2)
    {
        // Normal output
        os << "=== SQP Solver Statistics ===" << std::endl;
        os << "  Iterations:             " << number_of_iterations << std::endl;
        os << "  Total Cost:             " << total_cost << std::endl;
        os << "  Constraint Violation:   " << total_constraint_violation << std::endl;
        os << "  Dynamics Defect:        " << total_dynamics_defect << std::endl;
        os << "  Dual Infeasibility:     " << dual_infeasibility << std::endl;
        os << "  Step Norm:              " << step_norm << std::endl;
        os << "  Linesearch Iterations:  " << linesearch_iterations << std::endl;
        {
            const char* status_str[] = {"OK", "MAX_ITER", "MIN_STEP", "NaN"};
            const char* s = (qp_status >= 0 && qp_status <= 3) ? status_str[qp_status] : "UNKNOWN";
            os << "  QP info:                " << s << " in " << qp_iterations << " iters" << std::endl;
        }
        os << "  Last Iteration Time:    " << std::fixed << std::setprecision(3) << last_iteration_time_ms << " ms" << std::endl;
        os << "  Total Time:             " << std::setprecision(3) << total_time_ms / 1000.0 << " s" << std::endl;
    }
}

void SQPstatistics::update()
{
    auto now = std::chrono::high_resolution_clock::now();
    last_iteration_time_ms = std::chrono::duration<double, std::milli>(now - _iteration_start_time).count();
    total_time_ms = std::chrono::duration<double, std::milli>(now - _solve_start_time).count();
    _iteration_start_time = now;
    number_of_iterations++;
}

void SQPstatistics::start_timer()
{
    auto now = std::chrono::high_resolution_clock::now();
    _solve_start_time = now;
    _iteration_start_time = now;
}

void SQPstatistics::update_iterations(int iter)
{
    number_of_iterations = iter;
}

void SQPstatistics::update_cost(double cost)
{
    total_cost = cost;
}

void SQPstatistics::update_constraint_violation(double violation)
{
    total_constraint_violation = violation;
}

void SQPstatistics::update_dynamics_defect(double defect)
{
    total_dynamics_defect = defect;
}


void SQPstatistics::update_step_norm(double norm)
{
    step_norm = norm;
}

void SQPstatistics::update_dual_infeasibility(double dual_infeas)
{
    dual_infeasibility = dual_infeas;
}

void SQPstatistics::update_linesearch_iterations(int ls_iter)
{
    linesearch_iterations = ls_iter;
}

void SQPstatistics::update_qp_info(int status, int iters)
{
    qp_status = status;
    qp_iterations = iters;
}
