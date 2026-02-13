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
    linesearch_iterations = 0;
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
    os << std::scientific << std::setprecision(4);

    if (verbosity == 0)
    {
        // Normal output - header at beginning and every 10 iterations
        if (number_of_iterations == 1 || number_of_iterations % 10 == 0)
        {
            os << std::setw(6) << "Iter"
               << std::setw(16) << "Cost"
               << std::setw(16) << "Violation"
               << std::setw(16) << "Defect"
               << std::setw(16) << "StepNorm"
               << std::setw(10) << "LSiters"
               << std::setw(14) << "Time(ms)"
               << std::setw(14) << "TotTime(ms)" << std::endl;
        }
        os << std::setw(6) << number_of_iterations
           << std::setw(16) << total_cost
           << std::setw(16) << total_constraint_violation
           << std::setw(16) << total_dynamics_defect
           << std::setw(16) << step_norm
           << std::setw(10) << linesearch_iterations
           << std::fixed << std::setw(14) << std::setprecision(3) << last_iteration_time_ms
           << std::setw(14) << std::setprecision(1) << total_time_ms << std::endl;
    }
    else if (verbosity == 1)
    {
        // Normal output
        os << "=== SQP Solver Statistics ===" << std::endl;
        os << "  Iterations:             " << number_of_iterations << std::endl;
        os << "  Total Cost:             " << total_cost << std::endl;
        os << "  Constraint Violation:   " << total_constraint_violation << std::endl;
        os << "  Dynamics Defect:        " << total_dynamics_defect << std::endl;
        os << "  Step Norm:              " << step_norm << std::endl;
        os << "  Linesearch Iterations:  " << linesearch_iterations << std::endl;
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

void SQPstatistics::update_linesearch_iterations(int ls_iter)
{
    linesearch_iterations = ls_iter;
}
