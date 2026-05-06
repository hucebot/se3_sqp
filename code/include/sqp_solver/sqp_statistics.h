#pragma once

#include <fstream>
#include <string>
#include <chrono>
#include <vector>

struct SQPstatistics {
    int number_of_iterations;
    double total_cost;
    double total_constraint_violation;
    double total_dynamics_defect;
    double step_norm;
    double dual_infeasibility;    // KKT stationarity residual (gradient of Lagrangian)
    int linesearch_iterations;
    int qp_status;          // HPIPM convergence status (0=success, 1=max_iter, 2=min_step, 3=NaN)
    int qp_iterations;      // Number of IPM iterations in last QP solve

    std::vector<double> per_node_violation;  // constraint violation per node, k=0..N-1
    bool print_per_node_violation = false;   // set from SQPoptions; controls per-node print

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
    void update_dual_infeasibility(double dual_infeas);
    void update_linesearch_iterations(int ls_iter);
    void update_qp_info(int status, int iters);
    void update_per_node_violations(const std::vector<double>& violations);

    // Timing functions
    void start_timer();

   private:
    void print_internal(std::ostream& os, int verbosity) const;

    // Timer state
    std::chrono::high_resolution_clock::time_point _solve_start_time;
    std::chrono::high_resolution_clock::time_point _iteration_start_time;
};
