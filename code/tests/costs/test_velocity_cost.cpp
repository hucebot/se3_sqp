#include <gtest/gtest.h>
#include <trajopt/costs/velocity_cost.h>
#include <trajopt/node.h>
#include "pinocchio_fixtures.h"
#include "numerical_differentiation.h"

class VelocityCostTest : public DoubleIntegratorFixture {};

TEST_F(VelocityCostTest, ZeroResidualAtReference) {
    VectorXd v_ref = node->v();  // reference = current config (zero)
    auto cost = std::make_shared<VelocityCost>(v_ref);
    node->add_cost(cost);

    cost->evaluate();

    EXPECT_LT(cost->get_value().norm(), 1e-12);
}

TEST_F(VelocityCostTest, NonZeroResidualAwayFromReference) {
    VectorXd q_ref = node->v();  // reference at zero
    node->v()(0) = 1.0;          // perturb current config

    auto cost = std::make_shared<VelocityCost>(q_ref);
    node->add_cost(cost);

    cost->evaluate();

    EXPECT_GT(cost->get_value().norm(), 1e-6);
}

TEST_F(VelocityCostTest, JacobianMatchesFiniteDifferences) {
    const int nv = model.nv;

    // Non-trivial state
    node->q()(0) = 0.7;
    node->v()(0) = 0.3;

    VectorXd q_ref(model.nq);
    q_ref(0) = 0.2;

    auto cost = std::make_shared<VelocityCost>(q_ref);
    node->add_cost(cost);

    cost->jacobian();
    MatrixXd analytical_jac = cost->get_jac_x();

    // Numerical Jacobian via Lie-group perturbations (state only, no control)
    auto eval_func = [&]() -> VectorXd {
        cost->evaluate();
        return cost->get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian_node(
        eval_func, *node, nv, /*perturb_x=*/true, /*perturb_u=*/false);

    EXPECT_TRUE(jacobians_match(analytical_jac, numerical_jac, 1e-5, 1e-8));
}

TEST_F(VelocityCostTest, NoControlDependency) {
    VectorXd q_ref = node->q();
    auto cost = std::make_shared<VelocityCost>(q_ref);
    node->add_cost(cost);

    cost->jacobian();
    MatrixXd Ju = cost->get_jac_u();

    EXPECT_EQ(Ju.cols(), 0);
}
