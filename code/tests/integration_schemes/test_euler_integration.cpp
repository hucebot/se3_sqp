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
    node->q() = VectorXd::Zero(model.nq);
    node->v() = VectorXd::Zero(model.nv);
    node->u() = VectorXd::Zero(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    euler->set_node(node.get());

    VectorXd var;
    VectorXd output(2 * model.nv);

    euler->evaluate(var, output);

    // With zero velocity and acceleration:
    // - Position integrated = q + dt*0 = 0
    // - Velocity output = v + dt*a = 0
    EXPECT_NEAR(output(0), 0.0, 1e-10);
    EXPECT_NEAR(output(1), 0.0, 1e-10);
}

TEST_F(EulerEvaluateTest, ConstantVelocityZeroAcceleration) {
    node->q() = VectorXd::Zero(model.nq);
    node->q()(0) = 1.0;
    node->v() = VectorXd::Ones(model.nv);
    node->u() = VectorXd::Zero(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    euler->set_node(node.get());

    VectorXd var;
    VectorXd output(2 * model.nv);

    euler->evaluate(var, output);

    // Position: q_integrated = q + dt*v = 1 + 0.1*1 = 1.1
    // Velocity: v_integrated = v + dt*a = 1 + 0 = 1
    EXPECT_NEAR(output(0), 1.1, 1e-10);
    EXPECT_NEAR(output(1), 1.0, 1e-10);
}

TEST_F(EulerEvaluateTest, AccelerationOnly) {
    node->q() = VectorXd::Zero(model.nq);
    node->v() = VectorXd::Zero(model.nv);
    node->u() = VectorXd::Ones(model.nv) * 2.0;

    auto euler = std::make_shared<EulerIntegration>(dt);
    euler->set_node(node.get());

    VectorXd var;
    VectorXd output(2 * model.nv);

    euler->evaluate(var, output);

    // Position: q_integrated = q + dt*v = 0 + 0.1*0 = 0
    // Velocity: v_integrated = v + dt*a = 0 + 0.1*2 = 0.2
    EXPECT_NEAR(output(0), 0.0, 1e-10);
    EXPECT_NEAR(output(1), 0.2, 1e-10);
}

TEST_F(EulerEvaluateTest, CombinedMotion) {
    node->q() = VectorXd::Zero(model.nq);
    node->q()(0) = 2.0;
    node->v() = VectorXd::Ones(model.nv) * 3.0;
    node->u() = VectorXd::Ones(model.nv) * (-1.0);

    auto euler = std::make_shared<EulerIntegration>(dt);
    euler->set_node(node.get());

    VectorXd var;
    VectorXd output(2 * model.nv);

    euler->evaluate(var, output);

    // Position: 2.0 + 0.1 * 3.0 = 2.3
    // Velocity: 3.0 + 0.1 * (-1.0) = 2.9
    EXPECT_NEAR(output(0), 2.3, 1e-10);
    EXPECT_NEAR(output(1), 2.9, 1e-10);
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
    euler->set_node(node.get());

    VectorXd var;
    MatrixXd jac(2 * model.nv, 3 * model.nv);
    euler->jacobian(var, jac);

    // Jacobian should be 2*nv x 3*nv
    EXPECT_EQ(jac.rows(), 2 * model.nv);
    EXPECT_EQ(jac.cols(), 3 * model.nv);
}

TEST_F(EulerJacobianTest, JacobianMatchesNumerical_AtOrigin) {
    node->q() = VectorXd::Zero(model.nq);
    node->v() = VectorXd::Zero(model.nv);
    node->u() = VectorXd::Zero(model.nv);

    auto euler = std::make_shared<EulerIntegration>(dt);
    euler->set_node(node.get());

    VectorXd var;
    MatrixXd analytical_jac(2 * model.nv, 3 * model.nv);
    euler->jacobian(var, analytical_jac);

    // Numerical Jacobian via finite differences
    auto eval_func = [&](const VectorXd& state) -> VectorXd {
        node->q() = state.head(model.nq);
        node->v() = state.segment(model.nq, model.nv);
        node->u() = state.tail(model.nv);

        VectorXd output(2 * model.nv);
        euler->evaluate(var, output);
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

    auto euler = std::make_shared<EulerIntegration>(dt);
    euler->set_node(node.get());

    VectorXd var;
    MatrixXd analytical_jac(2 * model.nv, 3 * model.nv);
    euler->jacobian(var, analytical_jac);

    auto eval_func = [&](const VectorXd& state) -> VectorXd {
        node->q() = state.head(model.nq);
        node->v() = state.segment(model.nq, model.nv);
        node->u() = state.tail(model.nv);

        VectorXd output(2 * model.nv);
        euler->evaluate(var, output);
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
    euler->set_node(node.get());

    VectorXd var;
    MatrixXd jac(2 * model.nv, 3 * model.nv);
    euler->jacobian(var, jac);

    // Check velocity-acceleration block is dt*I
    // jac(nv:2*nv, 2*nv:3*nv) = dt * I
    MatrixXd expected_va_block = dt * MatrixXd::Identity(model.nv, model.nv);
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

    auto euler = std::make_shared<EulerIntegration>(large_dt);
    euler->set_node(node.get());

    VectorXd var;
    VectorXd output(2 * model.nv);

    euler->evaluate(var, output);

    EXPECT_NEAR(output(0), 1.0, 1e-10);  // 0 + 1*1
    EXPECT_NEAR(output(1), 2.0, 1e-10);  // 1 + 1*1
}

TEST_F(EulerEdgeCaseTest, VerySmallTimestep) {
    double small_dt = 1e-10;

    node->q() = VectorXd::Ones(model.nq);
    node->v() = VectorXd::Ones(model.nv);
    node->u() = VectorXd::Ones(model.nv);

    auto euler = std::make_shared<EulerIntegration>(small_dt);
    euler->set_node(node.get());

    VectorXd var;
    VectorXd output(2 * model.nv);

    euler->evaluate(var, output);

    // With tiny dt, output should be approximately [q, v]
    EXPECT_NEAR(output(0), 1.0, 1e-8);
    EXPECT_NEAR(output(1), 1.0, 1e-8);
}

TEST_F(EulerEdgeCaseTest, LargeValues) {
    node->q() = VectorXd::Ones(model.nq) * 1e6;
    node->v() = VectorXd::Ones(model.nv) * 1e3;
    node->u() = VectorXd::Ones(model.nv) * 1e2;

    auto euler = std::make_shared<EulerIntegration>(dt);
    euler->set_node(node.get());

    VectorXd var;
    VectorXd output(2 * model.nv);

    euler->evaluate(var, output);

    EXPECT_FALSE(std::isnan(output(0)));
    EXPECT_FALSE(std::isnan(output(1)));
    EXPECT_FALSE(std::isinf(output(0)));
    EXPECT_FALSE(std::isinf(output(1)));
}
