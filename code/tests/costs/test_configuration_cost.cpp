#include <gtest/gtest.h>
#include <trajopt/costs/configuration_cost.h>
#include <trajopt/node.h>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include "pinocchio_fixtures.h"
#include "numerical_differentiation.h"

class ConfigurationCostTest : public DoubleIntegratorFixture {};

TEST_F(ConfigurationCostTest, ZeroResidualAtReference) {
    // Set node to a random valid config, then use it as the reference
    node->q() = pinocchio::randomConfiguration(model);
    VectorXd q_ref = node->q();
    auto cost = std::make_shared<ConfigurationCost>(q_ref);
    node->add_cost(cost);

    cost->evaluate();

    EXPECT_LT(cost->get_value().norm(), 1e-12);
}

TEST_F(ConfigurationCostTest, NonZeroResidualAwayFromReference) {
    VectorXd q_ref = pinocchio::randomConfiguration(model);
    node->q() = pinocchio::randomConfiguration(model);

    auto cost = std::make_shared<ConfigurationCost>(q_ref);
    node->add_cost(cost);

    cost->evaluate();

    EXPECT_GT(cost->get_value().norm(), 1e-6);
}

TEST_F(ConfigurationCostTest, JacobianMatchesFiniteDifferences) {
    const int nv = model.nv;

    // Non-trivial state
    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);

    VectorXd q_ref = pinocchio::randomConfiguration(model);

    auto cost = std::make_shared<ConfigurationCost>(q_ref);
    node->add_cost(cost);

    cost->jacobian();
    MatrixXd analytical_jac = cost->get_jac_x();

    // Manifold-aware numerical Jacobian (state only, no control)
    auto eval_func = [&]() -> VectorXd {
        cost->evaluate();
        return cost->get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian_node(
        eval_func, *node, nv, /*perturb_x=*/true, /*perturb_u=*/false);

    EXPECT_TRUE(jacobians_match(analytical_jac, numerical_jac, 1e-5, 1e-8));
}

TEST_F(ConfigurationCostTest, NoControlDependency) {
    VectorXd q_ref = pinocchio::randomConfiguration(model);
    auto cost = std::make_shared<ConfigurationCost>(q_ref);
    node->add_cost(cost);

    cost->jacobian();
    MatrixXd Ju = cost->get_jac_u();

    EXPECT_EQ(Ju.cols(), 0);
}
