#include <gtest/gtest.h>
#include <trajopt/constraints/integration_schemes/semi-euler.h>
#include <trajopt/node.h>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include "pinocchio_fixtures.h"
#include "numerical_differentiation.h"

class SemiEulerIntegrationTest : public DoubleIntegratorFixture {};

TEST_F(SemiEulerIntegrationTest, ZeroResidualWhenDynamicsSatisfied) {
    const int nv = model.nv;

    // Random valid configuration and velocity
    VectorXd q0 = pinocchio::randomConfiguration(model);
    VectorXd v0 = VectorXd::Random(nv);
    VectorXd a0 = VectorXd::Random(nv);

    node->q() = q0;
    node->v() = v0;
    node->u() = a0;

    // Next state satisfies semi-implicit Euler exactly:
    //   v_{k+1} = v_k + dt * a_k
    //   q_{k+1} = q_k ⊕ (dt * v_{k+1})
    VectorXd v_next = v0 + dt * a0;
    VectorXd q_next(model.nq);
    pinocchio::integrate(model, q0, v_next * dt, q_next);
    next_node->v() = v_next;
    next_node->q() = q_next;

    auto euler = std::make_shared<SemiEulerIntegration>(dt);
    node->add_dynamics(euler);

    euler->evaluate();

    EXPECT_LT(euler->get_value().norm(), 1e-10);
}

TEST_F(SemiEulerIntegrationTest, JacobianMatchesFiniteDifferences) {
    const int nv = model.nv;

    // Random valid states for both nodes
    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);
    node->u() = VectorXd::Random(nv);
    next_node->q() = pinocchio::randomConfiguration(model);
    next_node->v() = VectorXd::Random(nv);

    auto euler = std::make_shared<SemiEulerIntegration>(dt);
    node->add_dynamics(euler);

    euler->evaluate();
    euler->jacobian();
    MatrixXd analytical_full(2 * nv, 3 * nv);
    analytical_full.leftCols(2 * nv) = euler->get_jac_x();
    analytical_full.rightCols(nv)    = euler->get_jac_u();

    // Manifold-aware FD on (x_k, u_k)
    auto eval_func = [&]() -> VectorXd {
        euler->evaluate();
        return euler->get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian_node(
        eval_func, *node, 2 * nv, /*perturb_x=*/true, /*perturb_u=*/true);

    EXPECT_TRUE(jacobians_match(analytical_full, numerical_jac, 1e-4, 1e-6));
}
