#include <gtest/gtest.h>
#include <trajopt/constraints/integration_schemes/euler.h>
#include <trajopt/node.h>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include "pinocchio_fixtures.h"
#include "numerical_differentiation.h"

// ============================================================================
// Test Suite: EulerIntegration Construction
// ============================================================================

TEST(EulerIntegrationConstruction, PositiveTimestep) {
    EulerIntegration euler(0.01);
    SUCCEED();
}

TEST(EulerIntegrationConstruction, ZeroTimestep) {
    EulerIntegration euler(0.0);
    SUCCEED();
}

// ============================================================================
// Test Suite: evaluate() Correctness
// ============================================================================

class EulerEvaluateTest : public DoubleIntegratorFixture {};

TEST_F(EulerEvaluateTest, ZeroVelocityZeroAcceleration) {
    // Set current and next states
    node->q() = VectorXd::Zero(model.nq);
    node->v() = VectorXd::Zero(model.nv);
    node->u() = VectorXd::Zero(model.nv);
    next_node->q() = VectorXd::Zero(model.nq);
    next_node->v() = VectorXd::Zero(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    node->add_dynamics(euler);

    VectorXd output(2 * model.nv);
    euler->evaluate(output);

    // Constraint residual should be zero when dynamics are satisfied:
    // g = [q_next - (q + dt*v), v_next - (v + dt*u)]
    // g = [0 - (0 + 0), 0 - (0 + 0)] = [0, 0]
    EXPECT_NEAR(output(0), 0.0, 1e-10);
    EXPECT_NEAR(output(1), 0.0, 1e-10);
}

TEST_F(EulerEvaluateTest, ConstantVelocityZeroAcceleration) {
    // Current state: q=1, v=1, u=0
    node->q()(0) = 1.0;
    node->v() = VectorXd::Ones(model.nv);
    node->u() = VectorXd::Zero(model.nv);

    // Next state satisfies dynamics: q_next = q + dt*v, v_next = v + dt*u
    next_node->q()(0) = 1.0 + dt * 1.0;  // 1.1
    next_node->v() = VectorXd::Ones(model.nv);  // v_next = v + 0 = 1

    auto euler = std::make_shared<EulerIntegration>(dt);
    node->add_dynamics(euler);

    VectorXd output(2 * model.nv);
    euler->evaluate(output);

    // Residual should be zero when dynamics are satisfied
    EXPECT_NEAR(output(0), 0.0, 1e-10);
    EXPECT_NEAR(output(1), 0.0, 1e-10);
}

TEST_F(EulerEvaluateTest, AccelerationOnly) {
    // Current state: q=0, v=0, u=2
    node->q() = VectorXd::Zero(model.nq);
    node->v() = VectorXd::Zero(model.nv);
    node->u() = VectorXd::Ones(model.nv) * 2.0;

    // Next state satisfies dynamics
    next_node->q() = VectorXd::Zero(model.nq);  // q_next = 0 + 0.1*0 = 0
    next_node->v() = VectorXd::Ones(model.nv) * 0.2;  // v_next = 0 + 0.1*2 = 0.2

    auto euler = std::make_shared<EulerIntegration>(dt);
    node->add_dynamics(euler);

    VectorXd output(2 * model.nv);
    euler->evaluate(output);

    // Residual should be zero when dynamics are satisfied
    EXPECT_NEAR(output(0), 0.0, 1e-10);
    EXPECT_NEAR(output(1), 0.0, 1e-10);
}

TEST_F(EulerEvaluateTest, CombinedMotion) {
    // Current state: q=2, v=3, u=-1
    node->q()(0) = 2.0;
    node->v() = VectorXd::Ones(model.nv) * 3.0;
    node->u() = VectorXd::Ones(model.nv) * (-1.0);

    // Next state satisfies dynamics
    next_node->q()(0) = 2.0 + dt * 3.0;  // 2.3
    next_node->v() = VectorXd::Ones(model.nv) * 2.9;  // 3.0 + 0.1*(-1.0) = 2.9

    auto euler = std::make_shared<EulerIntegration>(dt);
    node->add_dynamics(euler);

    VectorXd output(2 * model.nv);
    euler->evaluate(output);

    // Residual should be zero when dynamics are satisfied
    EXPECT_NEAR(output(0), 0.0, 1e-10);
    EXPECT_NEAR(output(1), 0.0, 1e-10);
}

// ============================================================================
// Test Suite: jacobian() Correctness
// ============================================================================

class EulerJacobianTest : public DoubleIntegratorFixture {};

TEST_F(EulerJacobianTest, JacobianDimensions) {
    node->q() = VectorXd::Random(model.nq);
    node->v() = VectorXd::Random(model.nv);
    node->u() = VectorXd::Random(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    node->add_dynamics(euler);

    MatrixXd jac(2 * model.nv, 3 * model.nv);
    euler->jacobian(jac);

    // Jacobian should be 2*nv x 3*nv
    EXPECT_EQ(jac.rows(), 2 * model.nv);
    EXPECT_EQ(jac.cols(), 3 * model.nv);
}

TEST_F(EulerJacobianTest, JacobianMatchesNumerical_AtOrigin) {
    node->q() = VectorXd::Zero(model.nq);
    node->v() = VectorXd::Zero(model.nv);
    node->u() = VectorXd::Zero(model.nv);
    next_node->q() = VectorXd::Zero(model.nq);
    next_node->v() = VectorXd::Zero(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    node->add_dynamics(euler);

    MatrixXd analytical_jac(2 * model.nv, 3 * model.nv);
    euler->jacobian(analytical_jac);

    // Numerical Jacobian via finite differences (holding next_node fixed)
    auto eval_func = [&](const VectorXd& state) -> VectorXd {
        node->q() = state.head(model.nq);
        node->v() = state.segment(model.nq, model.nv);
        node->u() = state.tail(model.nv);

        VectorXd output(2 * model.nv);
        euler->evaluate(output);
        return output;
    };

    VectorXd state(3 * model.nv);
    state << node->q(), node->v(), node->u();

    MatrixXd numerical_jac = numerical_jacobian(eval_func, state, 2 * model.nv);

    EXPECT_TRUE(jacobians_match(analytical_jac, numerical_jac, 1e-5, 1e-8))
        << "Analytical:\n" << analytical_jac << "\nNumerical:\n" << numerical_jac;
}

TEST_F(EulerJacobianTest, JacobianMatchesNumerical_NonzeroState) {
    node->q() = VectorXd::Random(model.nq);
    node->v() = VectorXd::Random(model.nv);
    node->u() = VectorXd::Random(model.nv);
    next_node->q() = VectorXd::Random(model.nq);
    next_node->v() = VectorXd::Random(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    node->add_dynamics(euler);

    MatrixXd analytical_jac(2 * model.nv, 3 * model.nv);
    euler->jacobian(analytical_jac);

    auto eval_func = [&](const VectorXd& state) -> VectorXd {
        node->q() = state.head(model.nq);
        node->v() = state.segment(model.nq, model.nv);
        node->u() = state.tail(model.nv);

        VectorXd output(2 * model.nv);
        euler->evaluate(output);
        return output;
    };

    VectorXd state(3 * model.nv);
    state << node->q(), node->v(), node->u();

    MatrixXd numerical_jac = numerical_jacobian(eval_func, state, 2 * model.nv);

    EXPECT_TRUE(jacobians_match(analytical_jac, numerical_jac, 1e-5, 1e-8));
}

TEST_F(EulerJacobianTest, JacobianStructure_VelocityAccelerationBlock) {
    node->q() = VectorXd::Random(model.nq);
    node->v() = VectorXd::Random(model.nv);
    node->u() = VectorXd::Random(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    node->add_dynamics(euler);

    MatrixXd jac(2 * model.nv, 3 * model.nv);
    euler->jacobian(jac);

    // Check velocity-acceleration block is -dt*I
    // Constraint: v_{k+1} - v_k - dt*u_k = 0, so ∂g/∂u_k = -dt*I
    // jac(nv:2*nv, 2*nv:3*nv) = -dt * I
    MatrixXd expected_va_block = -dt * MatrixXd::Identity(model.nv, model.nv);
    MatrixXd actual_va_block = jac.block(model.nv, 2*model.nv, model.nv, model.nv);

    EXPECT_TRUE(actual_va_block.isApprox(expected_va_block, 1e-10))
        << "Expected:\n" << expected_va_block << "\nActual:\n" << actual_va_block;
}

// ============================================================================
// Test Suite: Edge Cases
// ============================================================================

class EulerEdgeCaseTest : public DoubleIntegratorFixture {};

TEST_F(EulerEdgeCaseTest, LargeTimestep) {
    double large_dt = 1.0;

    node->q() = VectorXd::Zero(model.nq);
    node->v() = VectorXd::Ones(model.nv);
    node->u() = VectorXd::Ones(model.nv);

    // Set next state to satisfy dynamics
    next_node->q() = VectorXd::Ones(model.nq);  // 0 + 1*1 = 1
    next_node->v() = VectorXd::Ones(model.nv) * 2.0;  // 1 + 1*1 = 2

    auto euler = std::make_shared<EulerIntegration>(large_dt);
    node->add_dynamics(euler);

    VectorXd output(2 * model.nv);
    euler->evaluate(output);

    // Residual should be zero when dynamics are satisfied
    EXPECT_NEAR(output(0), 0.0, 1e-10);
    EXPECT_NEAR(output(1), 0.0, 1e-10);
}

TEST_F(EulerEdgeCaseTest, VerySmallTimestep) {
    double small_dt = 1e-10;

    node->q() = VectorXd::Ones(model.nq);
    node->v() = VectorXd::Ones(model.nv);
    node->u() = VectorXd::Ones(model.nv);

    // With tiny dt, integrated state ≈ current state
    next_node->q() = VectorXd::Ones(model.nq);  // 1 + tiny*1 ≈ 1
    next_node->v() = VectorXd::Ones(model.nv);  // 1 + tiny*1 ≈ 1

    auto euler = std::make_shared<EulerIntegration>(small_dt);
    node->add_dynamics(euler);

    VectorXd output(2 * model.nv);
    euler->evaluate(output);

    // Residual should be very small (approximately zero)
    EXPECT_NEAR(output(0), 0.0, 1e-8);
    EXPECT_NEAR(output(1), 0.0, 1e-8);
}

TEST_F(EulerEdgeCaseTest, LargeValues) {
    node->q() = VectorXd::Ones(model.nq) * 1e6;
    node->v() = VectorXd::Ones(model.nv) * 1e3;
    node->u() = VectorXd::Ones(model.nv) * 1e2;

    // Set next state to satisfy dynamics with large values
    next_node->q() = VectorXd::Ones(model.nq) * (1e6 + dt * 1e3);
    next_node->v() = VectorXd::Ones(model.nv) * (1e3 + dt * 1e2);

    auto euler = std::make_shared<EulerIntegration>(dt);
    node->add_dynamics(euler);

    VectorXd output(2 * model.nv);
    euler->evaluate(output);

    // Check no numerical issues with large values
    EXPECT_FALSE(std::isnan(output(0)));
    EXPECT_FALSE(std::isnan(output(1)));
    EXPECT_FALSE(std::isinf(output(0)));
    EXPECT_FALSE(std::isinf(output(1)));
}

// ============================================================================
// Test Suite: New Jacobian Interface (Internal Storage)
// ============================================================================

class EulerJacobianNewInterfaceTest : public ::testing::Test {
protected:
    pinocchio::Model model;
    std::shared_ptr<Node> node_k;
    std::shared_ptr<Node> node_k_next;
    double dt = 0.1;

    // Trajectory storage for nodes
    VectorXd x_k_storage, u_k_storage;
    VectorXd x_k_next_storage, u_k_next_storage;

    void SetUp() override {
        // Create simple 1-DOF model
        model.name = "double_integrator";
        auto joint_id = model.addJoint(
            0,
            pinocchio::JointModelPX(),
            pinocchio::SE3::Identity(),
            "slider"
        );
        model.addJointFrame(joint_id);
        model.lowerPositionLimit.resize(1);
        model.upperPositionLimit.resize(1);
        model.lowerPositionLimit << -10.0;
        model.upperPositionLimit << 10.0;

        // Create two nodes (current and next)
        node_k = std::make_shared<Node>(model);
        node_k_next = std::make_shared<Node>(model);
        node_k->next_node = node_k_next.get();

        // Allocate trajectory storage
        x_k_storage.setZero(node_k->nx());
        u_k_storage.setZero(node_k->nu());
        x_k_next_storage.setZero(node_k_next->nx());
        u_k_next_storage.setZero(node_k_next->nu());

        // Bind nodes to storage
        node_k->bind_trajectory(&x_k_storage, &u_k_storage);
        node_k_next->bind_trajectory(&x_k_next_storage, &u_k_next_storage);
    }

    void TearDown() override {
        node_k.reset();
        node_k_next.reset();
    }
};

TEST_F(EulerJacobianNewInterfaceTest, NoArgsJacobianExecutes) {
    node_k->q() = VectorXd::Random(model.nq);
    node_k->v() = VectorXd::Random(model.nv);
    node_k->u() = VectorXd::Random(model.nv);
    node_k_next->q() = VectorXd::Random(model.nq);
    node_k_next->v() = VectorXd::Random(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    node_k->add_dynamics(euler);

    // Call jacobian() with no arguments - should not crash
    EXPECT_NO_THROW(euler->jacobian());
}

TEST_F(EulerJacobianNewInterfaceTest, GetJacXDimensions) {
    node_k->q() = VectorXd::Random(model.nq);
    node_k->v() = VectorXd::Random(model.nv);
    node_k->u() = VectorXd::Random(model.nv);
    node_k_next->q() = VectorXd::Random(model.nq);
    node_k_next->v() = VectorXd::Random(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    node_k->add_dynamics(euler);
    euler->jacobian();

    MatrixXd jac_x = euler->get_jac_x();

    // ∂g/∂x_k should be (2*nv × 2*nv)
    EXPECT_EQ(jac_x.rows(), 2 * model.nv);
    EXPECT_EQ(jac_x.cols(), 2 * model.nv);
}

TEST_F(EulerJacobianNewInterfaceTest, GetJacUDimensions) {
    node_k->q() = VectorXd::Random(model.nq);
    node_k->v() = VectorXd::Random(model.nv);
    node_k->u() = VectorXd::Random(model.nv);
    node_k_next->q() = VectorXd::Random(model.nq);
    node_k_next->v() = VectorXd::Random(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    node_k->add_dynamics(euler);
    euler->jacobian();

    MatrixXd jac_u = euler->get_jac_u();

    // ∂g/∂u_k should be (2*nv × nv)
    EXPECT_EQ(jac_u.rows(), 2 * model.nv);
    EXPECT_EQ(jac_u.cols(), model.nv);
}

TEST_F(EulerJacobianNewInterfaceTest, JacUStructure) {
    node_k->q() = VectorXd::Random(model.nq);
    node_k->v() = VectorXd::Random(model.nv);
    node_k->u() = VectorXd::Random(model.nv);
    node_k_next->q() = VectorXd::Random(model.nq);
    node_k_next->v() = VectorXd::Random(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    node_k->add_dynamics(euler);
    euler->jacobian();

    MatrixXd jac_u = euler->get_jac_u();

    // Structure of ∂g/∂u_k:
    // Top half (position constraint): should be zero (position doesn't depend on u)
    // Bottom half (velocity constraint): should be -dt*I

    // Check top half is zero
    MatrixXd top_half = jac_u.topRows(model.nv);
    EXPECT_TRUE(top_half.isZero(1e-10))
        << "Top half of jac_u should be zero:\n" << top_half;

    // Check bottom half is -dt*I
    MatrixXd bottom_half = jac_u.bottomRows(model.nv);
    MatrixXd expected_bottom = -dt * MatrixXd::Identity(model.nv, model.nv);
    EXPECT_TRUE(bottom_half.isApprox(expected_bottom, 1e-10))
        << "Expected:\n" << expected_bottom << "\nActual:\n" << bottom_half;
}

TEST_F(EulerJacobianNewInterfaceTest, JacXVelocityBlockStructure) {
    node_k->q() = VectorXd::Random(model.nq);
    node_k->v() = VectorXd::Random(model.nv);
    node_k->u() = VectorXd::Random(model.nv);
    node_k_next->q() = VectorXd::Random(model.nq);
    node_k_next->v() = VectorXd::Random(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    node_k->add_dynamics(euler);
    euler->jacobian();

    MatrixXd jac_x = euler->get_jac_x();

    // For velocity constraint: ∂(v_{k+1} - v_k - dt*u_k)/∂v_k = -I
    MatrixXd vel_wrt_v_block = jac_x.block(model.nv, model.nv, model.nv, model.nv);
    MatrixXd expected = -MatrixXd::Identity(model.nv, model.nv);

    EXPECT_TRUE(vel_wrt_v_block.isApprox(expected, 1e-10))
        << "Expected:\n" << expected << "\nActual:\n" << vel_wrt_v_block;
}

TEST_F(EulerJacobianNewInterfaceTest, ConsistencyBetweenInterfaces) {
    node_k->q() = VectorXd::Random(model.nq);
    node_k->v() = VectorXd::Random(model.nv);
    node_k->u() = VectorXd::Random(model.nv);
    node_k_next->q() = VectorXd::Random(model.nq);
    node_k_next->v() = VectorXd::Random(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    node_k->add_dynamics(euler);

    // Call new interface
    euler->jacobian();
    MatrixXd jac_x_new = euler->get_jac_x();
    MatrixXd jac_u_new = euler->get_jac_u();

    // Call legacy interface
    MatrixXd jac_legacy(2 * model.nv, 3 * model.nv);
    euler->jacobian(jac_legacy);

    // Compare blocks: legacy [∂g/∂q_k | ∂g/∂v_k | ∂g/∂u_k]
    // New: get_jac_x() gives [∂g/∂q_k | ∂g/∂v_k]
    //      get_jac_u() gives ∂g/∂u_k

    MatrixXd jac_x_from_legacy = jac_legacy.leftCols(2 * model.nv);
    MatrixXd jac_u_from_legacy = jac_legacy.rightCols(model.nv);

    EXPECT_TRUE(jac_x_new.isApprox(jac_x_from_legacy, 1e-10))
        << "get_jac_x() should match legacy interface";

    EXPECT_TRUE(jac_u_new.isApprox(jac_u_from_legacy, 1e-10))
        << "get_jac_u() should match legacy interface";
}
