#include <gtest/gtest.h>
#include <trajopt/constraints/friction_cone_constraint.h>
#include <trajopt/node.h>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include "../fixtures/pinocchio_fixtures.h"
#include "../utils/numerical_differentiation.h"

class FrictionConeConstraintJacobianTest : public DoubleIntegratorFixture {};

TEST_F(FrictionConeConstraintJacobianTest, JacobianMatchesFiniteDifferencesActive) {
    const int nv = model.nv;

    // Register contact
    node->add_contact("contact_1");

    // Update storage sizes and rebind
    u_storage.resize(node->nu());
    u_storage.setZero();
    u_next_storage.resize(next_node->nu());
    u_next_storage.setZero();
    node->bind_trajectory(&x_storage, &u_storage);
    next_node->bind_trajectory(&x_next_storage, &u_next_storage);

    // Non-trivial random state
    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);

    // Set non-trivial contact force (within friction cone)
    double mu = 0.7;
    node->fc(0) << 0.2, 0.3, 1.0;  // (fx, fy, fz) in local frame

    // Create friction cone constraint (contact is active by default)
    auto friction_constraint = std::make_shared<FrictionConeConstraint>("contact_1", mu);
    node->add_constraint(friction_constraint);

    // Analytical Jacobian
    friction_constraint->evaluate();
    friction_constraint->jacobian();
    MatrixXd analytical_jac_x = friction_constraint->get_jac_x();
    MatrixXd analytical_jac_u = friction_constraint->get_jac_u();

    // Numerical Jacobian w.r.t. state x
    auto eval_func_x = [&]() -> VectorXd {
        friction_constraint->evaluate();
        return friction_constraint->get_value();
    };

    MatrixXd numerical_jac_x = numerical_jacobian_node(
        eval_func_x, *node, 5, /*perturb_x=*/true, /*perturb_u=*/false);

    // Numerical Jacobian w.r.t. control u
    auto eval_func_u = [&]() -> VectorXd {
        friction_constraint->evaluate();
        return friction_constraint->get_value();
    };

    MatrixXd numerical_jac_u = numerical_jacobian_node(
        eval_func_u, *node, 5, /*perturb_x=*/false, /*perturb_u=*/true);

    // Check state Jacobian
    if (!jacobians_match(analytical_jac_x, numerical_jac_x, 1e-4, 1e-6)) {
        std::cout << "State Jacobian Mismatch:\n";
        std::cout << "Analytical:\n" << analytical_jac_x << "\n\n";
        std::cout << "Numerical:\n" << numerical_jac_x << "\n\n";
        std::cout << "Difference:\n" << (analytical_jac_x - numerical_jac_x) << "\n\n";
    }

    // Check control Jacobian
    if (!jacobians_match(analytical_jac_u, numerical_jac_u, 1e-4, 1e-6)) {
        std::cout << "Control Jacobian Mismatch:\n";
        std::cout << "Analytical:\n" << analytical_jac_u << "\n\n";
        std::cout << "Numerical:\n" << numerical_jac_u << "\n\n";
        std::cout << "Difference:\n" << (analytical_jac_u - numerical_jac_u) << "\n\n";
    }

    EXPECT_TRUE(jacobians_match(analytical_jac_x, numerical_jac_x, 1e-4, 1e-6));
    EXPECT_TRUE(jacobians_match(analytical_jac_u, numerical_jac_u, 1e-4, 1e-6));
}

TEST_F(FrictionConeConstraintJacobianTest, JacobianMatchesFiniteDifferencesInactive) {
    const int nv = model.nv;

    // Register contact and set inactive
    node->add_contact("contact_1");
    node->set_active_contacts({});  // No active contacts

    // Update storage sizes and rebind
    u_storage.resize(node->nu());
    u_storage.setZero();
    u_next_storage.resize(next_node->nu());
    u_next_storage.setZero();
    node->bind_trajectory(&x_storage, &u_storage);
    next_node->bind_trajectory(&x_next_storage, &u_next_storage);

    // Non-trivial random state
    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);

    // Set contact force (should not matter when inactive)
    node->fc(0) << 0.5, -0.3, 0.8;

    // Create friction cone constraint
    double mu = 0.6;
    auto friction_constraint = std::make_shared<FrictionConeConstraint>("contact_1", mu);
    node->add_constraint(friction_constraint);

    // Analytical Jacobian
    friction_constraint->evaluate();
    friction_constraint->jacobian();
    MatrixXd analytical_jac_x = friction_constraint->get_jac_x();
    MatrixXd analytical_jac_u = friction_constraint->get_jac_u();

    // Numerical Jacobian w.r.t. state x
    auto eval_func_x = [&]() -> VectorXd {
        friction_constraint->evaluate();
        return friction_constraint->get_value();
    };

    MatrixXd numerical_jac_x = numerical_jacobian_node(
        eval_func_x, *node, 5, /*perturb_x=*/true, /*perturb_u=*/false);

    // Numerical Jacobian w.r.t. control u
    auto eval_func_u = [&]() -> VectorXd {
        friction_constraint->evaluate();
        return friction_constraint->get_value();
    };

    MatrixXd numerical_jac_u = numerical_jacobian_node(
        eval_func_u, *node, 5, /*perturb_x=*/false, /*perturb_u=*/true);

    EXPECT_TRUE(jacobians_match(analytical_jac_x, numerical_jac_x, 1e-4, 1e-6));
    EXPECT_TRUE(jacobians_match(analytical_jac_u, numerical_jac_u, 1e-4, 1e-6));
}

TEST_F(FrictionConeConstraintJacobianTest, BoundsUpdateWithContactState) {
    const int nv = model.nv;

    // Register contact
    node->add_contact("contact_1");

    // Update storage sizes and rebind
    u_storage.resize(node->nu());
    u_storage.setZero();
    u_next_storage.resize(next_node->nu());
    u_next_storage.setZero();
    node->bind_trajectory(&x_storage, &u_storage);
    next_node->bind_trajectory(&x_next_storage, &u_next_storage);

    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);
    node->fc(0) << 0.1, 0.1, 1.0;

    auto friction_constraint = std::make_shared<FrictionConeConstraint>("contact_1", 0.5);
    node->add_constraint(friction_constraint);

    // Test active contact
    node->set_active_contacts({"contact_1"});
    friction_constraint->evaluate();
    VectorXd lower_active = friction_constraint->get_lower_bound();
    VectorXd upper_active = friction_constraint->get_upper_bound();

    EXPECT_TRUE(lower_active.isApprox(VectorXd::Zero(5)));
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(upper_active(i), std::numeric_limits<double>::infinity());
    }

    // Test inactive contact
    node->set_active_contacts({});
    friction_constraint->evaluate();
    VectorXd lower_inactive = friction_constraint->get_lower_bound();
    VectorXd upper_inactive = friction_constraint->get_upper_bound();

    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(lower_inactive(i), -std::numeric_limits<double>::infinity());
        EXPECT_EQ(upper_inactive(i), std::numeric_limits<double>::infinity());
    }
}
