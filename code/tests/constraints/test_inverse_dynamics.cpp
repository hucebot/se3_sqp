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

    // Analytical Jacobian
    inv_dyn->jacobian();
    MatrixXd analytical_jac = inv_dyn->get_jac_x();
    // Full Jacobian [∂g/∂x | ∂g/∂u] = [∂g/∂q | ∂g/∂v | ∂g/∂u]
    MatrixXd analytical_full(nv, 3 * nv);
    analytical_full.leftCols(2 * nv) = inv_dyn->get_jac_x();
    analytical_full.rightCols(nv) = inv_dyn->get_jac_u();

    // Input vector [q, v, u] for finite differences
    VectorXd x(3 * nv);
    x.head(nv)        = node->q();
    x.segment(nv, nv) = node->v();
    x.tail(nv)        = node->u();

    auto eval_func = [&](const VectorXd& input) -> VectorXd {
        node->q() = input.head(nv);
        node->v() = input.segment(nv, nv);
        node->u() = input.tail(nv);
        inv_dyn->evaluate();
        return inv_dyn->get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian(eval_func, x, nv);

    std::cout<<"analytical"<<std::endl<<analytical_full<<std::endl;
    std::cout<<"numerical"<<std::endl<<numerical_jac<<std::endl;

    EXPECT_TRUE(jacobians_match(analytical_full, numerical_jac, 1e-5, 1e-8));
}
