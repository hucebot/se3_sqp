#pragma once

#include <fstream>
#include <string>
#include <chrono>

struct SQPstatistics {
    int number_of_iterations;
    double total_cost;
    double total_constraint_violation;
    double total_dynamics_defect;
    double step_norm;
    int linesearch_iterations;

    // Timing
    double total_time_ms;
    double last_iteration_time_ms;

    // Constructor - initialize all values
    SQPstatistics();

    // Reset all statistics to default values
    void reset();

    // Print statistics with verbosity level (0=minimal, 1=normal, 2=detailed)
    void print(int verbosity = 0) const;

    // Print statistics to file
    void print_to_file(const std::string& filename, int verbosity = 1) const;

    // Update functions
    void update();
    void update_iterations(int iter);
    void update_cost(double cost);
    void update_constraint_violation(double violation);
    void update_dynamics_defect(double defect);
    void update_step_norm(double norm);
    void update_linesearch_iterations(int ls_iter);

    // Timing functions
    void start_timer();

   private:
    void print_internal(std::ostream& os, int verbosity) const;

    // Timer state
    std::chrono::high_resolution_clock::time_point _solve_start_time;
    std::chrono::high_resolution_clock::time_point _iteration_start_time;
};
