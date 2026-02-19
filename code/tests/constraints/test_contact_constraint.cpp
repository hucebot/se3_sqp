#include <gtest/gtest.h>
#include <trajopt/constraints/contact_constraint.h>
#include <trajopt/node.h>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include "../fixtures/pinocchio_fixtures.h"
#include "../utils/numerical_differentiation.h"

class ContactConstraintJacobianTest : public DoubleIntegratorFixture {};

TEST_F(ContactConstraintJacobianTest, JacobianMatchesFiniteDifferencesActive) {
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

    // Create contact constraint (contact is active by default)
    auto contact_constraint = std::make_shared<ContactConstraint>("contact_1", 0.0);
    node->add_constraint(contact_constraint);

    // Analytical Jacobian
    contact_constraint->evaluate();
    contact_constraint->jacobian();
    MatrixXd analytical_jac = contact_constraint->get_jac_x();

    // Manifold-aware FD on x_k only (no control dependency)
    auto eval_func = [&]() -> VectorXd {
        contact_constraint->evaluate();
        return contact_constraint->get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian_node(
        eval_func, *node, 4, /*perturb_x=*/true, /*perturb_u=*/false);

    if (!jacobians_match(analytical_jac, numerical_jac, 1e-4, 1e-6)) {
        std::cout << "Analytical Jacobian:\n" << analytical_jac << "\n\n";
        std::cout << "Numerical Jacobian:\n" << numerical_jac << "\n\n";
        std::cout << "Difference:\n" << (analytical_jac - numerical_jac) << "\n\n";
    }

    EXPECT_TRUE(jacobians_match(analytical_jac, numerical_jac, 1e-4, 1e-6));
}

TEST_F(ContactConstraintJacobianTest, JacobianMatchesFiniteDifferencesInactive) {
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

    // Create contact constraint (contact is now inactive)
    auto contact_constraint = std::make_shared<ContactConstraint>("contact_1", 0.0);
    node->add_constraint(contact_constraint);

    // Analytical Jacobian
    contact_constraint->evaluate();
    contact_constraint->jacobian();
    MatrixXd analytical_jac = contact_constraint->get_jac_x();

    // Manifold-aware FD on x_k only
    auto eval_func = [&]() -> VectorXd {
        contact_constraint->evaluate();
        return contact_constraint->get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian_node(
        eval_func, *node, 4, /*perturb_x=*/true, /*perturb_u=*/false);

    EXPECT_TRUE(jacobians_match(analytical_jac, numerical_jac, 1e-4, 1e-6));
}
