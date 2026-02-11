#include <gtest/gtest.h>
#include <trajopt/constraints/inverse_dynamics.h>
#include <trajopt/node.h>
#include "pinocchio_fixtures.h"
#include "numerical_differentiation.h"

class InvDynamicsJacobianTest : public DoubleIntegratorFixture {};

TEST_F(InvDynamicsJacobianTest, JacobianMatchesFiniteDifferences) {
    const int nv = model.nv;

    // Non-trivial state to exercise all Jacobian terms
    node->q()(0) = 1.0;
    node->v()(0) = 0.5;
    node->u()(0) = 2.0;

    auto inv_dyn = std::make_shared<InvDynamics>();
    node->add_constraint(inv_dyn);

    // Analytical Jacobian: nv x 3*nv
    MatrixXd analytical_jac(nv, 3 * nv);
    inv_dyn->jacobian(analytical_jac);

    // Input vector [q, v, u] for finite differences
    VectorXd x(3 * nv);
    x.head(nv)             = node->q();
    x.segment(nv, nv)      = node->v();
    x.tail(nv)             = node->u();

    auto eval_func = [&](const VectorXd& input) -> VectorXd {
        node->q() = input.head(nv);
        node->v() = input.segment(nv, nv);
        node->u() = input.tail(nv);
        VectorXd output(nv);
        inv_dyn->evaluate(output);
        return output;
    };

    MatrixXd numerical_jac = numerical_jacobian(eval_func, x, nv);

    EXPECT_TRUE(jacobians_match(analytical_jac, numerical_jac, 1e-5, 1e-8));
}
