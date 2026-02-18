#include <gtest/gtest.h>
#include <trajopt/constraints/inverse_dynamics.h>
#include <trajopt/node.h>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include "../fixtures/pinocchio_fixtures.h"
#include "../utils/numerical_differentiation.h"

class InvDynamicsJacobianTest : public DoubleIntegratorFixture {};

TEST_F(InvDynamicsJacobianTest, JacobianMatchesFiniteDifferences) {
    const int nv = model.nv;

    // Non-trivial random state
    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);
    node->u() = VectorXd::Random(nv);

    auto inv_dyn = std::make_shared<InvDynamics>();
    node->add_constraint(inv_dyn);

    // Analytical Jacobian
    inv_dyn->evaluate();  // Must evaluate first to update kinematics
    inv_dyn->jacobian();
    MatrixXd analytical_full(nv, 3 * nv);
    analytical_full.leftCols(2 * nv) = inv_dyn->get_jac_x();
    analytical_full.rightCols(nv) = inv_dyn->get_jac_u();

    // Manifold-aware FD on (x_k, u_k)
    auto eval_func = [&]() -> VectorXd {
        inv_dyn->evaluate();
        return inv_dyn->get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian_node(
        eval_func, *node, nv, /*perturb_x=*/true, /*perturb_u=*/true);

    EXPECT_TRUE(jacobians_match(analytical_full, numerical_jac, 1e-4, 1e-6));
}

TEST_F(InvDynamicsJacobianTest, WithSingleContactForce) {
    const int nv = model.nv;

    // Register the end_effector contact (frame already exists in fixture)
    node->add_contact("end_effector");

    // Update storage sizes for expanded control and REBIND
    u_storage.resize(node->nu());
    u_storage.setZero();
    u_next_storage.resize(next_node->nu());
    u_next_storage.setZero();
    node->bind_trajectory(&x_storage, &u_storage);
    next_node->bind_trajectory(&x_next_storage, &u_next_storage);

    // Non-trivial random state and control (acceleration + contact force)
    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);
    node->a() = VectorXd::Random(nv);
    node->fc(0) = VectorXd::Random(3);  // Random 3D contact force

    auto inv_dyn = std::make_shared<InvDynamics>();
    node->add_constraint(inv_dyn);

    // Analytical Jacobian
    inv_dyn->evaluate();  // Must evaluate first to update kinematics
    inv_dyn->jacobian();
    MatrixXd analytical_full(nv, 2*nv + nv + 3);
    analytical_full.leftCols(2 * nv) = inv_dyn->get_jac_x();
    analytical_full.rightCols(nv + 3) = inv_dyn->get_jac_u();

    // Manifold-aware FD on (x_k, u_k)
    auto eval_func = [&]() -> VectorXd {
        inv_dyn->evaluate();
        return inv_dyn->get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian_node(
        eval_func, *node, nv, /*perturb_x=*/true, /*perturb_u=*/true);

    // std::cout << "Analytical:\n" << analytical_full << "\n\n";
    // std::cout << "Numerical:\n" << numerical_jac << "\n\n";

    EXPECT_TRUE(jacobians_match(analytical_full, numerical_jac, 1e-4, 1e-6));
}

TEST_F(InvDynamicsJacobianTest, WithMultipleContactForces) {
    const int nv = model.nv;

    // Register contacts in specific order (frames already exist in fixture)
    node->add_contacts({"contact_1", "contact_2"});

    // Update storage sizes and REBIND
    u_storage.resize(node->nu());
    u_storage.setZero();
    u_next_storage.resize(next_node->nu());
    u_next_storage.setZero();
    node->bind_trajectory(&x_storage, &u_storage);
    next_node->bind_trajectory(&x_next_storage, &u_next_storage);

    // Non-trivial random state
    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);
    node->a() = VectorXd::Random(nv);
    node->fc(0) = VectorXd::Random(3);
    node->fc(1) = VectorXd::Random(3);

    auto inv_dyn = std::make_shared<InvDynamics>();
    node->add_constraint(inv_dyn);

    // Analytical Jacobian
    inv_dyn->evaluate();  // Must evaluate first to update kinematics
    inv_dyn->jacobian();
    MatrixXd analytical_full(nv, 2*nv + nv + 6);
    analytical_full.leftCols(2 * nv) = inv_dyn->get_jac_x();
    analytical_full.rightCols(nv + 6) = inv_dyn->get_jac_u();

    // Manifold-aware FD
    auto eval_func = [&]() -> VectorXd {
        inv_dyn->evaluate();
        return inv_dyn->get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian_node(
        eval_func, *node, nv, /*perturb_x=*/true, /*perturb_u=*/true);

    // std::cout << "Analytical:\n" << analytical_full << "\n\n";
    // std::cout << "Numerical:\n" << numerical_jac << "\n\n";

    EXPECT_TRUE(jacobians_match(analytical_full, numerical_jac, 1e-4, 1e-6));
}

TEST_F(InvDynamicsJacobianTest, InactiveContactsProduceZeroJacobian) {
    const int nv = model.nv;

    // Register both contacts (frames already exist in fixture)
    node->add_contacts({"contact_1", "contact_2"});

    // Activate only the first contact
    node->set_active_contacts({"contact_1"});

    // Update storage and REBIND
    u_storage.resize(node->nu());
    u_storage.setZero();
    node->bind_trajectory(&x_storage, &u_storage);

    // Set random state and forces
    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);
    node->a() = VectorXd::Random(nv);
    node->fc(0) = VectorXd::Random(3);  // active
    node->fc(1) = VectorXd::Random(3);  // inactive, should not affect torques

    auto inv_dyn = std::make_shared<InvDynamics>();
    node->add_constraint(inv_dyn);

    // Evaluate and get Jacobian
    inv_dyn->evaluate();
    inv_dyn->jacobian();
    MatrixXd jac_u = inv_dyn->get_jac_u();

    // Check that inactive contact (contact_2, columns nv+3 to nv+5) has zero Jacobian
    MatrixXd inactive_jac = jac_u.rightCols(3);
    EXPECT_TRUE(inactive_jac.isZero(1e-12));

    // Check that active contact (contact_1, columns nv to nv+2) has non-zero Jacobian
    MatrixXd active_jac = jac_u.middleCols(nv, 3);
    EXPECT_GT(active_jac.norm(), 1e-6);
}

TEST_F(InvDynamicsJacobianTest, InactiveContactForceDoesNotAffectTorques) {
    const int nv = model.nv;

    // Register contact (frame already exists in fixture)
    node->add_contact("contact_1");
    u_storage.resize(node->nu());
    u_storage.setZero();
    node->bind_trajectory(&x_storage, &u_storage);

    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);
    node->a() = VectorXd::Random(nv);

    auto inv_dyn = std::make_shared<InvDynamics>();
    node->add_constraint(inv_dyn);

    // Evaluate with contact active and zero force
    node->set_active_contacts({"contact_1"});
    node->fc(0).setZero();
    inv_dyn->evaluate();
    VectorXd tau_active_zero_force = inv_dyn->get_value();

    // Evaluate with contact inactive and non-zero force (should be ignored)
    node->set_active_contacts({});  // deactivate all
    node->fc(0) = VectorXd::Random(3);  // set random force
    inv_dyn->evaluate();
    VectorXd tau_inactive = inv_dyn->get_value();

    // Torques should be identical (inactive force is ignored)
    EXPECT_TRUE(tau_active_zero_force.isApprox(tau_inactive, 1e-12));
}
