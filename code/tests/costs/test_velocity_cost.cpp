#include <gtest/gtest.h>
#include <trajopt/costs/velocity_cost.h>
#include <trajopt/node.h>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include "pinocchio_fixtures.h"
#include "numerical_differentiation.h"

class VelocityCostTest : public DoubleIntegratorFixture {};

TEST_F(VelocityCostTest, ZeroResidualAtReference) {
    VectorXd v_ref = node->v();  // reference = current velocity (zero)
    auto cost = std::make_shared<VelocityCost>(v_ref.tail(node->nv()-6));
    node->add_cost(cost);

    cost->evaluate();

    EXPECT_LT(cost->get_value().norm(), 1e-12);
}

TEST_F(VelocityCostTest, NonZeroResidualAwayFromReference) {
    VectorXd v_ref = node->v();  // reference at zero
    node->v() = VectorXd::Random(model.nv);

    auto cost = std::make_shared<VelocityCost>(v_ref.tail(node->nv()-6));
    node->add_cost(cost);

    cost->evaluate();

    EXPECT_GT(cost->get_value().norm(), 1e-6);
}

TEST_F(VelocityCostTest, JacobianMatchesFiniteDifferences) {
    const int nv = model.nv;

    // Non-trivial state
    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);

    VectorXd v_ref = VectorXd::Random(nv-6);

    auto cost = std::make_shared<VelocityCost>(v_ref);
    node->add_cost(cost);

    cost->jacobian();
    MatrixXd analytical_jac = cost->get_jac_x();

    // Manifold-aware numerical Jacobian (state only, no control)
    auto eval_func = [&]() -> VectorXd {
        cost->evaluate();
        return cost->get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian_node(
        eval_func, *node, nv-6, /*perturb_x=*/true, /*perturb_u=*/false);

    EXPECT_TRUE(jacobians_match(analytical_jac, numerical_jac, 1e-5, 1e-8));
}

TEST_F(VelocityCostTest, NoControlDependency) {
    VectorXd v_ref = VectorXd::Random(model.nv-6);
    auto cost = std::make_shared<VelocityCost>(v_ref);
    node->add_cost(cost);

    cost->jacobian();
    MatrixXd Ju = cost->get_jac_u();

    // Ju should have columns (ndu) but all values should be zero (no control dependency)
    EXPECT_EQ(Ju.rows(), model.nv - 6);
    EXPECT_EQ(Ju.cols(), node->ndu());
    EXPECT_LT(Ju.norm(), 1e-12);
}
