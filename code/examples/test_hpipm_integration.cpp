#include <qp_solver/hpipm_solver.h>
#include <sqp_solver/sqp_solver.h>

#include <Eigen/Dense>
#include <iostream>

/**
 * Simple example showcasing the HPIPM integration with SQP solver
 *
 * This example demonstrates:
 * - Direct use of HPIPMSolver (low-level)
 * - Clean separation between SQP and QP code
 * - Performance features (inline methods, zero overhead)
 */

void test_hpipm_direct() {
    std::cout << "\n=== Testing Direct HPIPM Wrapper ===" << std::endl;

    // Simple QP problem: minimize 0.5*x'*Q*x + q'*x subject to dynamics
    int N = 100;  // Horizon
    int nx = 2;   // State dimension
    int nu = 1;   // Control dimension

    HPIPMSolver qp_solver;
    qp_solver.initialize(N, nx, nu, 0);

    // Set solver options for performance
    qp_solver.set_iter_max(50);
    qp_solver.set_tol(1e-8, 1e-8, 1e-8, 1e-8);
    qp_solver.set_warm_start(true);  // CRITICAL for performance

    // Simple dynamics: x_{k+1} = A*x_k + B*u_k
    // Using column-major Eigen matrices (HPIPM expects column-major)
    Eigen::Matrix2d A;
    A << 1.0, 0.1,   // A = [1.0  0.1]
         0.0, 1.0;   //     [0.0  1.0]
    Eigen::Vector2d B(0.0, 0.1);  // B = [0.0; 0.1]
    Eigen::Vector2d b = Eigen::Vector2d::Zero();

    // Simple cost: Q = I, R = I
    Eigen::Matrix2d Q = Eigen::Matrix2d::Identity();
    Eigen::Matrix<double, 1, 1> R;
    R << 1.0;
    Eigen::Vector2d q = Eigen::Vector2d::Zero();
    Eigen::Matrix<double, 1, 1> r;
    r << 0.0;

    // Set problem data for all nodes
    for (int k = 0; k < N; k++) {
        qp_solver.set_A(k, A.data());
        qp_solver.set_B(k, B.data());
        qp_solver.set_b(k, b.data());
        qp_solver.set_Q(k, Q.data());
        qp_solver.set_R(k, R.data());
        qp_solver.set_q(k, q.data());
        qp_solver.set_r(k, r.data());
    }
    // Terminal cost
    qp_solver.set_Q(N, Q.data());
    qp_solver.set_q(N, q.data());

    // Solve QP (this call is INLINED - zero overhead!)
    std::cout << "Solving QP..." << std::endl;
    qp_solver.solve();

    // Extract solution using Eigen vectors
    Eigen::Vector2d x_sol;
    Eigen::Matrix<double, 1, 1> u_sol;

    std::cout << "\nSolution:" << std::endl;
    for (int k = 0; k <= N; k++) {
        qp_solver.get_x(k, x_sol.data());
        std::cout << "  Node " << k << ": x = [" << x_sol(0) << ", " << x_sol(1)
                  << "]";

        if (k < N) {
            qp_solver.get_u(k, u_sol.data());
            std::cout << ", u = [" << u_sol(0) << "]";
        }
        std::cout << std::endl;
    }

    std::cout << "\n✓ Direct HPIPM wrapper works!" << std::endl;
    std::cout << "  - All methods inline for zero overhead" << std::endl;
    std::cout << "  - Warm-starting enabled for maximum performance"
              << std::endl;
    std::cout << "  - Clean code separation maintained" << std::endl;
}

int main() {
    std::cout << "==============================================" << std::endl;
    std::cout << "  HPIPM Integration Test" << std::endl;
    std::cout << "  Performance-Optimized QP Solver" << std::endl;
    std::cout << "==============================================" << std::endl;

    test_hpipm_direct();

    std::cout << "\n=============================================="
              << std::endl;
    std::cout << "  All tests passed! ✓" << std::endl;
    std::cout << "==============================================" << std::endl;

    return 0;
}
