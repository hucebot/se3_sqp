#include <gtest/gtest.h>
#include <trajopt/constraints/joint_limits_constraint.h>
#include <trajopt/node.h>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include "../fixtures/pinocchio_fixtures.h"
#include "../utils/numerical_differentiation.h"

class JointLimitsConstraintTest : public DoubleIntegratorFixture {};

TEST_F(JointLimitsConstraintTest, ConstraintDimensions) {
    auto joint_limits = std::make_shared<JointLimitsConstraint>();
    node->add_constraint(joint_limits);

    // Constraint dimension should equal 2*nv (position + velocity in tangent space)
    EXPECT_EQ(joint_limits->get_output_dim(), 2 * model.nv);
    EXPECT_EQ(joint_limits->get_input_dim(), node->ndx() + node->ndu());
}

TEST_F(JointLimitsConstraintTest, EvaluationReturnsStateComponents) {
    const int nv = model.nv;

    // Set random state
    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);

    auto joint_limits = std::make_shared<JointLimitsConstraint>();
    node->add_constraint(joint_limits);

    joint_limits->evaluate();

    // Constraint value dimension should be 2*nv
    EXPECT_EQ(joint_limits->get_value().size(), 2 * nv);

    // Position part is q in tangent space (difference from neutral config)
    VectorXd expected_q_tangent(nv);
    pinocchio::difference(model, pinocchio::neutral(model), node->q(), expected_q_tangent);
    EXPECT_TRUE(joint_limits->get_value().head(nv).isApprox(expected_q_tangent));

    // Velocity part is just v
    EXPECT_TRUE(joint_limits->get_value().tail(nv).isApprox(node->v()));
}

TEST_F(JointLimitsConstraintTest, JacobianMatchesFiniteDifferences) {
    const int nv = model.nv;

    // Set non-trivial random state
    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);

    auto joint_limits = std::make_shared<JointLimitsConstraint>();
    node->add_constraint(joint_limits);

    // Analytical Jacobian
    joint_limits->evaluate();
    joint_limits->jacobian();
    MatrixXd analytical_jac_x = joint_limits->get_jac_x();
    MatrixXd analytical_jac_u = joint_limits->get_jac_u();

    // Numerical Jacobian w.r.t. state x
    auto eval_func_x = [&]() -> VectorXd {
        joint_limits->evaluate();
        return joint_limits->get_value();
    };

    MatrixXd numerical_jac_x = numerical_jacobian_node(
        eval_func_x, *node, 2 * nv, /*perturb_x=*/true, /*perturb_u=*/false);

    // Numerical Jacobian w.r.t. control u
    auto eval_func_u = [&]() -> VectorXd {
        joint_limits->evaluate();
        return joint_limits->get_value();
    };

    MatrixXd numerical_jac_u = numerical_jacobian_node(
        eval_func_u, *node, 2 * nv, /*perturb_x=*/false, /*perturb_u=*/true);

    // Check state Jacobian
    if (!jacobians_match(analytical_jac_x, numerical_jac_x, 1e-4, 1e-6)) {
        std::cout << "State Jacobian Mismatch:\n";
        std::cout << "Analytical:\n" << analytical_jac_x << "\n\n";
        std::cout << "Numerical:\n" << numerical_jac_x << "\n\n";
        std::cout << "Difference:\n" << (analytical_jac_x - numerical_jac_x) << "\n\n";
    }

    // Check control Jacobian (should be zero)
    if (!jacobians_match(analytical_jac_u, numerical_jac_u, 1e-4, 1e-6)) {
        std::cout << "Control Jacobian Mismatch:\n";
        std::cout << "Analytical:\n" << analytical_jac_u << "\n\n";
        std::cout << "Numerical:\n" << numerical_jac_u << "\n\n";
        std::cout << "Difference:\n" << (analytical_jac_u - numerical_jac_u) << "\n\n";
    }

    EXPECT_TRUE(jacobians_match(analytical_jac_x, numerical_jac_x, 1e-4, 1e-6));
    EXPECT_TRUE(jacobians_match(analytical_jac_u, numerical_jac_u, 1e-4, 1e-6));
}

TEST_F(JointLimitsConstraintTest, JacobianWrtControlIsZero) {
    const int nv = model.nv;

    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);

    auto joint_limits = std::make_shared<JointLimitsConstraint>();
    node->add_constraint(joint_limits);

    joint_limits->evaluate();
    joint_limits->jacobian();

    MatrixXd jac_u = joint_limits->get_jac_u();

    // Since g(q) doesn't depend on u, Jacobian w.r.t. u should be zero
    EXPECT_TRUE(jac_u.isZero(1e-12));
}

TEST_F(JointLimitsConstraintTest, ConstraintViolationDetection) {
    const int nv = model.nv;

    auto joint_limits = std::make_shared<JointLimitsConstraint>();
    node->add_constraint(joint_limits);

    VectorXd lower = joint_limits->get_lower_bound();
    VectorXd upper = joint_limits->get_upper_bound();

    // Test with neutral configuration and zero velocity (should be feasible)
    node->q() = pinocchio::neutral(model);
    node->v() = VectorXd::Zero(nv);
    joint_limits->evaluate();
    VectorXd value = joint_limits->get_value();

    // Position part should be zero (neutral config)
    EXPECT_TRUE(value.head(nv).isZero(1e-10));

    // Velocity part should be zero
    EXPECT_TRUE(value.tail(nv).isZero(1e-10));

    // Both should be within bounds
    for (int i = 0; i < 2 * nv; ++i) {
        EXPECT_GE(value(i), lower(i)) << "Failed at index " << i;
        EXPECT_LE(value(i), upper(i)) << "Failed at index " << i;
    }
}
