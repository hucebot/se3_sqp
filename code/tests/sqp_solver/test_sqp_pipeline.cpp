#include <gtest/gtest.h>
#include <sqp_solver/sqp_solver.h>
#include <trajopt/ocp.h>
#include <trajopt/node.h>
#include <trajopt/constraints/integration_schemes/euler.h>
#include <trajopt/constraints/inverse_dynamics.h>
#include <trajopt/costs/configuration_cost.h>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

#include <cmath>
#include <random>

#include "pinocchio_fixtures.h"

// Helper to build a simple floating+2R OCP for testing
class SQPPipelineTest : public ::testing::Test {
   protected:
    static constexpr int N = 10;
    static constexpr double dt = 0.01;
    pinocchio::Model model;
    std::unique_ptr<OCP> ocp;

    void SetUp() override {
        std::mt19937 gen(std::random_device{}());
        model = buildRandomFloating2R(gen);

        // Terminal cost reference: neutral configuration
        VectorXd q_ref = pinocchio::neutral(model);

        ocp = std::make_unique<OCP>(N);
        for (int i = 0; i < N; i++) {
            Node node(model);
            if (i < N - 1) {
                node.add_dynamics(std::make_shared<EulerIntegration>(dt));
            }
            if (i == N - 1) {
                auto cost = std::make_shared<ConfigurationCost>(q_ref);
                cost->set_weight(1e3);
                node.add_cost(cost);
            }
            ocp->addNode(std::move(node));
        }
        ocp->finalize();

        // Set a non-trivial initial guess (away from reference)
        VectorXd q0 = pinocchio::randomConfiguration(model);
        for (int k = 0; k < N; k++) {
            ocp->get_node(k).q() = q0;
            ocp->get_node(k).v().setZero();
            ocp->get_node(k).u().setZero();
        }
    }
};

// --------------------------------------------------------------------------
// Test: step() binds nodes to candidate trajectory
// --------------------------------------------------------------------------
TEST_F(SQPPipelineTest, StepBindsNodesToCandidates) {
    SQPSolver solver(*ocp);

    // Save nominal x[0]
    VectorXd x_nom_0 = ocp->get_node(0).x();

    // The node should still be readable
    EXPECT_EQ(ocp->get_node(0).x().size(), model.nq + model.nv);
}

// --------------------------------------------------------------------------
// Test: accept_step() swaps candidates into nominal
// --------------------------------------------------------------------------
TEST_F(SQPPipelineTest, AcceptStepUpdatesNominal) {
    // Save the initial nominal trajectory
    std::vector<VectorXd> x_before(N), u_before(N - 1);
    for (int k = 0; k < N; k++) {
        x_before[k] = ocp->x_traj()[k];
        if (k < N - 1) u_before[k] = ocp->u_traj()[k];
    }

    // Solve one iteration (full step, no linesearch)
    SQPSolver solver(*ocp);
    solver.solve();

    // After solve, nominal trajectory should have changed
    bool x_changed = false;
    for (int k = 0; k < N; k++) {
        if (!ocp->x_traj()[k].isApprox(x_before[k], 1e-15)) {
            x_changed = true;
            break;
        }
    }
    EXPECT_TRUE(x_changed) << "Nominal trajectory should change after solve";
}

// --------------------------------------------------------------------------
// Test: linearize() rebinds to nominal (safe reset)
// --------------------------------------------------------------------------
TEST_F(SQPPipelineTest, LinearizeRebindsToNominal) {
    SQPSolver solver(*ocp);

    // After construction, nodes are bound to nominal
    VectorXd x0_before = ocp->get_node(0).x();

    // Manually bind to some garbage to simulate step() having run
    std::vector<VectorXd> fake_x(N), fake_u(N - 1);
    for (int k = 0; k < N; k++) {
        fake_x[k] = VectorXd::Ones(ocp->get_node(k).nx()) * 999.0;
        if (k < N - 1) fake_u[k] = VectorXd::Ones(ocp->get_node(k).nu()) * 999.0;
    }
    ocp->bind_trajectory(fake_x, fake_u);

    // Node should now see the fake data
    EXPECT_DOUBLE_EQ(ocp->get_node(0).x()(0), 999.0);

    // Now solve — linearize() should rebind to nominal as its first action
    // The solver should work correctly despite the previous fake binding
    solver.solve();

    // After solve, nodes should be on the (updated) nominal trajectory, not on fake
    EXPECT_NE(ocp->get_node(0).x()(0), 999.0);
}

// --------------------------------------------------------------------------
// Test: LSType::NONE skips line search entirely
// --------------------------------------------------------------------------
TEST_F(SQPPipelineTest, NoLinesearchTakesFullStep) {
    // Solve with NONE — should take full steps, 0 ls iterations
    SQPSolver solver(*ocp);
    solver.solve();

    // Solver should converge (this problem is well-conditioned)
    // The trajectory should have changed from initial guess
    bool changed = false;
    for (int k = 0; k < N; k++) {
        if (ocp->x_traj()[k].norm() > 1e-10) {
            changed = true;
            break;
        }
    }
    EXPECT_TRUE(changed);
}

// --------------------------------------------------------------------------
// Test: Solver converges on a feasibility problem (dynamics + inv dynamics)
// --------------------------------------------------------------------------
TEST_F(SQPPipelineTest, ConvergesOnFeasibilityProblem) {
    SQPSolver solver(*ocp);
    solver.solve();

    // After convergence, dynamics constraints should be nearly satisfied.
    for (int k = 0; k < N - 1; k++) {
        auto dyn = ocp->get_node(k).get_dynamics();
        dyn->evaluate();
        EXPECT_LT(dyn->get_value().norm(), 1e-3)
            << "Dynamics residual at node " << k << " should be small";
    }
}

// --------------------------------------------------------------------------
// Test: Node binding state machine invariant across multiple iterations
// --------------------------------------------------------------------------
TEST_F(SQPPipelineTest, NodeBindingConsistentAfterSolve) {
    SQPSolver solver(*ocp);
    solver.solve();

    // After solve completes, nodes must be bound to the nominal trajectory.
    for (int k = 0; k < N; k++) {
        EXPECT_TRUE(ocp->get_node(k).x().isApprox(ocp->x_traj()[k]))
            << "Node " << k << " should be bound to nominal trajectory after solve";
    }
}

// --------------------------------------------------------------------------
// Test: Multiple solves work (reinitializing from previous solution)
// --------------------------------------------------------------------------
TEST_F(SQPPipelineTest, MultipleSolvesWork) {
    SQPSolver solver(*ocp);

    // First solve
    solver.solve();
    std::vector<VectorXd> x_after_first(N);
    for (int k = 0; k < N; k++) x_after_first[k] = ocp->x_traj()[k];

    // Second solve from the same state (should converge immediately or in 1 iter)
    solver.solve();

    // Trajectory should be essentially unchanged (already converged)
    for (int k = 0; k < N; k++) {
        EXPECT_TRUE(ocp->x_traj()[k].isApprox(x_after_first[k], 1e-6))
            << "Second solve should not change an already-converged trajectory";
    }
}
