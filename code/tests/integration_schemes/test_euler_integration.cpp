#include <gtest/gtest.h>
#include <trajopt/constraints/integration_schemes/euler.h>
#include <trajopt/node.h>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include "pinocchio_fixtures.h"
#include "numerical_differentiation.h"

class EulerIntegrationTest : public DoubleIntegratorFixture {};

TEST_F(EulerIntegrationTest, ZeroResidualWhenDynamicsSatisfied) {
    node->q()(0) = 1.0;
    node->v()(0) = 2.0;
    node->u()(0) = 3.0;

    // Next state satisfies Euler dynamics exactly
    next_node->q()(0) = 1.0 + dt * 2.0;
    next_node->v()(0) = 2.0 + dt * 3.0;

    auto euler = std::make_shared<EulerIntegration>(dt);
    node->add_dynamics(euler);

    euler->evaluate();

    EXPECT_LT(euler->get_value().norm(), 1e-10);
}

TEST_F(EulerIntegrationTest, JacobianMatchesFiniteDifferences) {
    const int nv = model.nv;

    node->q()(0) = 1.0;
    node->v()(0) = 0.5;
    node->u()(0) = 2.0;
    next_node->q()(0) = 0.3;
    next_node->v()(0) = 0.7;

    auto euler = std::make_shared<EulerIntegration>(dt);
    node->add_dynamics(euler);

    euler->jacobian();
    MatrixXd analytical_full(2 * nv, 3 * nv);
    analytical_full.leftCols(2 * nv) = euler->get_jac_x();
    analytical_full.rightCols(nv)    = euler->get_jac_u();

    VectorXd x(3 * nv);
    x.head(nv)        = node->q();
    x.segment(nv, nv) = node->v();
    x.tail(nv)        = node->u();

    auto eval_func = [&](const VectorXd& input) -> VectorXd {
        node->q() = input.head(nv);
        node->v() = input.segment(nv, nv);
        node->u() = input.tail(nv);
        euler->evaluate();
        return euler->get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian(eval_func, x, 2 * nv);

    EXPECT_TRUE(jacobians_match(analytical_full, numerical_jac, 1e-5, 1e-8));
}
