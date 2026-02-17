#include <gtest/gtest.h>
#include <trajopt/constraints/inverse_dynamics.h>
#include <trajopt/node.h>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include "pinocchio_fixtures.h"
#include "numerical_differentiation.h"

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
