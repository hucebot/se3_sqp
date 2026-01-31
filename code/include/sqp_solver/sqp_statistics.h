#pragma once

#include <fstream>
#include <string>

struct SQPstatistics {
    int number_of_iterations;
    double total_cost;
    double total_constraint_violation;
    double step_norm;
    double linesearch_iterations;

    // Constructor - initialize all values
    SQPstatistics();

    // Reset all statistics to default values
    void reset();

    // Print statistics with verbosity level (0=minimal, 1=normal, 2=detailed)
    void print(int verbosity = 1) const;

    // Print statistics to file
    void print_to_file(const std::string& filename, int verbosity = 1) const;

    // Update functions
    void update_iterations(int iter);
    void update_cost(double cost);
    void update_constraint_violation(double violation);
    void update_step_norm(double norm);
    void update_linesearch_iterations(double ls_iter);

   private:
    void print_internal(std::ostream& os, int verbosity) const;
};
