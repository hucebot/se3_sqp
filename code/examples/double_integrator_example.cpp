#include <qp_solver/hpipm_solver.h>
#include <sqp_solver/sqp_solver.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

/**
 * Double Integrator Example
 *
 * This example demonstrates trajectory optimization for a double integrator
 * system.
 *
 * System dynamics:
 *   - State: x = [position, velocity]
 *   - Control: u = [acceleration]
 *   - Continuous dynamics:
 *       position_dot = velocity
 *       velocity_dot = acceleration
 *   - Discretized dynamics (Euler integration):
 *       x_{k+1} = A*x_k + B*u_k
 *       where A = [1  dt]    B = [0.5*dt^2]
 *                 [0   1]        [dt      ]
 *
 * Objective:
 *   - Move from initial position to target position
 *   - Minimize control effort
 *   - Satisfy velocity and control limits
 */

void run_double_integrator_example() {
    std::cout << "\n=== Double Integrator Trajectory Optimization ==="
              << std::endl;

    // Problem setup
    const int N =
        30;  // Number of shooting nodes (reduced for better conditioning)
    const int nx = 2;         // State dimension [position, velocity]
    const int nu = 1;         // Control dimension [acceleration]
    const double dt = 0.1;    // Time step (seconds)
    const double T = N * dt;  // Total time horizon

    std::cout << "\nProblem parameters:" << std::endl;
    std::cout << "  Horizon length: N = " << N << " nodes" << std::endl;
    std::cout << "  Time step: dt = " << dt << " s" << std::endl;
    std::cout << "  Total time: T = " << T << " s" << std::endl;
    std::cout << "  State dimension: nx = " << nx << " [position, velocity]"
              << std::endl;
    std::cout << "  Control dimension: nu = " << nu << " [acceleration]"
              << std::endl;

    // Initial and target states
    Eigen::Vector2d x_init(0.0, 0.0);    // Start at rest at origin
    Eigen::Vector2d x_target(5.0, 0.0);  // Move to position 5 m, come to rest

    std::cout << "\nInitial state: [" << x_init.transpose() << "]" << std::endl;
    std::cout << "Target state:  [" << x_target.transpose() << "]" << std::endl;

    // Box constraint limits
    const double v_max = 5.0;   // Max velocity [m/s]
    const double a_max = 10.0;  // Max acceleration [m/s^2]
    const int nbx =
        nx;  // State box constraints (initial state + velocity limits)
    const int nbu = nu;  // Control box constraints (acceleration limits)

    // Initialize QP solver
    // Parameters: N (horizon), nx (state dim), nu (control dim), ng (general
    // constraints), nbx (state box constraints), nbu (control box constraints)
    HPIPMSolver qp_solver;
    qp_solver.initialize(N, nx, nu, 0, nbx, nbu);

    // Solver options
    qp_solver.set_iter_max(100);
    qp_solver.set_tol(1e-8, 1e-8, 1e-8, 1e-8);
    qp_solver.set_warm_start(true);

    // Box constraint indices (which variables are constrained)
    // State: [position, velocity] - constrain both at stage 0, velocity only
    // elsewhere
    std::vector<int> idxbx(nbx);
    for (int i = 0; i < nbx; i++) {
        idxbx[i] = i;  // Constrain indices 0, 1, ..., nbx-1
    }

    // Control: [acceleration] - constrain all controls
    std::vector<int> idxbu(nbu);
    for (int i = 0; i < nbu; i++) {
        idxbu[i] = i;
    }

    // State bounds
    const double inf = 1e20;            // Large number for "unbounded"
    Eigen::Vector2d lbx_init = x_init;  // Initial state (hard constraint)
    Eigen::Vector2d ubx_init = x_init;
    Eigen::Vector2d lbx_stage(-inf, -v_max);  // Position free, velocity bounded
    Eigen::Vector2d ubx_stage(inf, v_max);

    // Control bounds
    Eigen::Matrix<double, 1, 1> lbu, ubu;
    lbu << -a_max;
    ubu << a_max;

    // Set box constraints for all stages
    for (int k = 0; k <= N; k++) {
        // State box constraint indices
        qp_solver.set_idxbx(k, idxbx.data());

        if (k == 0) {
            // Initial state: hard constraint x(0) = x_init
            qp_solver.set_lbx(k, lbx_init.data());
            qp_solver.set_ubx(k, ubx_init.data());
        } else {
            // Other stages: velocity limits only
            qp_solver.set_lbx(k, lbx_stage.data());
            qp_solver.set_ubx(k, ubx_stage.data());
        }

        // Control bounds (stages 0 to N-1)
        if (k < N) {
            qp_solver.set_idxbu(k, idxbu.data());
            qp_solver.set_lbu(k, lbu.data());
            qp_solver.set_ubu(k, ubu.data());
        }
    }

    std::cout << "\nBox constraints:" << std::endl;
    std::cout << "  Initial state: x(0) = [" << x_init.transpose()
              << "] (hard constraint)" << std::endl;
    std::cout << "  Velocity limit: |v| <= " << v_max << " m/s" << std::endl;
    std::cout << "  Acceleration limit: |a| <= " << a_max << " m/s^2"
              << std::endl;

    // Discretized dynamics matrices (Euler integration)
    // x_{k+1} = x_k + dt * v_k
    // v_{k+1} = v_k + dt * a_k
    // Column-major layout for HPIPM compatibility
    Eigen::Matrix2d A;
    A << 1.0, dt,  // A = [1  dt]
        0.0, 1.0;  //     [0   1]

    Eigen::Vector2d B;
    B << 0.0,  // B = [0 ]  (position not directly affected by acceleration)
        dt;    //     [dt]  (velocity changes with acceleration)

    Eigen::Vector2d b = Eigen::Vector2d::Zero();  // No affine term

    std::cout << "\nDiscretized dynamics (Euler integration):" << std::endl;
    std::cout << "  x_{k+1} = A*x_k + B*u_k" << std::endl;
    std::cout << "  A = [" << A(0, 0) << "  " << A(0, 1) << "]" << std::endl;
    std::cout << "      [" << A(1, 0) << "  " << A(1, 1) << "]" << std::endl;
    std::cout << "  B = [" << B(0) << "]" << std::endl;
    std::cout << "      [" << B(1) << "]" << std::endl;

    // Cost matrices
    // Stage cost: only minimize control effort (NO state tracking)
    double R_acc = 1.0;  // Control effort weight

    Eigen::Matrix2d Q =
        Eigen::Matrix2d::Zero();  // No state cost at intermediate stages
    Eigen::Matrix<double, 1, 1> R;
    R << R_acc;
    Eigen::Vector2d q = Eigen::Vector2d::Zero();  // No linear state cost
    Eigen::Matrix<double, 1, 1> r;
    r << 0.0;  // No linear control cost

    // Terminal cost: reach target state
    double Q_pos_terminal = 1e6;
    double Q_vel_terminal = 1e3;

    Eigen::Matrix2d Q_N;
    Q_N << Q_pos_terminal, 0.0, 0.0, Q_vel_terminal;
    Eigen::Vector2d q_N;
    q_N << -Q_pos_terminal * x_target(0), -Q_vel_terminal * x_target(1);

    std::cout << "\nCost function:" << std::endl;
    std::cout << "  Stage cost: minimize control effort only (R=" << R_acc
              << ")" << std::endl;
    std::cout << "  Terminal cost: reach target [" << x_target.transpose()
              << "] (Q=" << Q_pos_terminal << ")" << std::endl;

    // Set problem data for all stages
    for (int k = 0; k < N; k++) {
        // Dynamics
        qp_solver.set_A(k, A.data());
        qp_solver.set_B(k, B.data());
        qp_solver.set_b(k, b.data());

        // Stage cost (same for all stages - initial state enforced via box
        // constraint)
        qp_solver.set_Q(k, Q.data());
        qp_solver.set_q(k, q.data());

        // Control cost
        qp_solver.set_R(k, R.data());
        qp_solver.set_r(k, r.data());
    }

    // Terminal cost
    qp_solver.set_Q(N, Q_N.data());
    qp_solver.set_q(N, q_N.data());

    // Solve the QP
    std::cout << "\nSolving trajectory optimization problem..." << std::endl;
    qp_solver.solve();
    std::cout << "Optimization complete!" << std::endl;

    // Extract and display solution using Eigen vectors
    Eigen::Vector2d x_sol;
    Eigen::Matrix<double, 1, 1> u_sol;

    std::cout << "\n=== Optimal Trajectory ===" << std::endl;
    std::cout
        << "Time [s] | Position [m] | Velocity [m/s] | Acceleration [m/s^2]"
        << std::endl;
    std::cout
        << "---------|--------------|----------------|---------------------"
        << std::endl;

    // Track statistics
    double max_vel = 0.0;
    double max_acc = 0.0;
    double total_control_effort = 0.0;

    // Print first and last few steps, compute all statistics
    const int print_first = 5;
    const int print_last = 5;

    for (int k = 0; k <= N; k++) {
        qp_solver.get_x(k, x_sol.data());

        double time = k * dt;
        double pos = x_sol(0);
        double vel = x_sol(1);

        max_vel = std::max(max_vel, std::abs(vel));

        // Only print first few, last few, and markers
        bool should_print = (k < print_first) || (k > N - print_last);

        if (k == print_first && N > print_first + print_last) {
            std::cout << "   ...   |     ...      |      ...       |       ..."
                      << std::endl;
        }

        if (should_print) {
            printf("  %5.2f  |   %8.4f   |    %8.4f    |", time, pos, vel);

            if (k < N) {
                qp_solver.get_u(k, u_sol.data());
                double acc = u_sol(0);
                printf("      %8.4f\n", acc);
            } else {
                printf("         ---\n");
            }
        }

        // Always compute statistics
        if (k < N) {
            qp_solver.get_u(k, u_sol.data());
            double acc = u_sol(0);
            max_acc = std::max(max_acc, std::abs(acc));
            total_control_effort += acc * acc * dt;
        }
    }

    // Final statistics
    qp_solver.get_x(N, x_sol.data());
    Eigen::Vector2d x_final = x_sol;
    Eigen::Vector2d tracking_error = x_final - x_target;

    std::cout << "\n=== Solution Statistics ===" << std::endl;
    std::cout << "  Final position:        " << x_final(0)
              << " m (target: " << x_target(0) << " m)" << std::endl;
    std::cout << "  Final velocity:        " << x_final(1)
              << " m/s (target: " << x_target(1) << " m/s)" << std::endl;
    std::cout << "  Tracking error:        [" << tracking_error.transpose()
              << "]" << std::endl;
    std::cout << "  Tracking error norm:   " << tracking_error.norm()
              << std::endl;
    std::cout << "  Max velocity:          " << max_vel << " m/s" << std::endl;
    std::cout << "  Max acceleration:      " << max_acc << " m/s^2"
              << std::endl;
    std::cout << "  Total control effort:  " << total_control_effort
              << " m^2/s^3" << std::endl;

    std::cout << "\n=== Example Complete ===" << std::endl;
}

int main() {
    std::cout << "=============================================" << std::endl;
    std::cout << "  Double Integrator Example" << std::endl;
    std::cout << "  Trajectory Optimization Demo" << std::endl;
    std::cout << "=============================================" << std::endl;

    run_double_integrator_example();

    std::cout << "\n=============================================" << std::endl;
    std::cout << "  Example finished successfully!" << std::endl;
    std::cout << "=============================================" << std::endl;

    return 0;
}
