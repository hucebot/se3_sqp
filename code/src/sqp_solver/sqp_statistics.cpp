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
    step_norm = 0.0;
    linesearch_iterations = 0.0;
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
    os << std::fixed << std::setprecision(6);

    if (verbosity == 0)
    {
        // Minimal output - single line
        os << "Iter: " << number_of_iterations
           << " | Cost: " << total_cost
           << " | Viol: " << total_constraint_violation << std::endl;
    }
    else if (verbosity == 1)
    {
        // Normal output
        os << "=== SQP Solver Statistics ===" << std::endl;
        os << "  Iterations:             " << number_of_iterations << std::endl;
        os << "  Total Cost:             " << total_cost << std::endl;
        os << "  Constraint Violation:   " << total_constraint_violation << std::endl;
        os << "  Step Norm:              " << step_norm << std::endl;
        os << "  Linesearch Iterations:  " << linesearch_iterations << std::endl;
    }
    else
    {
        // Detailed output
        os << "=====================================" << std::endl;
        os << "  SQP SOLVER STATISTICS (DETAILED)   " << std::endl;
        os << "=====================================" << std::endl;
        os << "  Number of Iterations:        " << number_of_iterations << std::endl;
        os << "  Total Cost:                  " << total_cost << std::endl;
        os << "  Total Constraint Violation:  " << total_constraint_violation << std::endl;
        os << "  Final Step Norm:             " << step_norm << std::endl;
        os << "  Linesearch Iterations:       " << linesearch_iterations << std::endl;
        os << "=====================================" << std::endl;
    }
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

void SQPstatistics::update_step_norm(double norm)
{
    step_norm = norm;
}

void SQPstatistics::update_linesearch_iterations(double ls_iter)
{
    linesearch_iterations = ls_iter;
}
