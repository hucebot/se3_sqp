#include <gtest/gtest.h>
#include <trajopt/costs/configuration_cost.h>
#include <trajopt/node.h>
#include "pinocchio_fixtures.h"
#include "numerical_differentiation.h"

class ConfigurationCostTest : public DoubleIntegratorFixture {};

TEST_F(ConfigurationCostTest, ZeroResidualAtReference) {
    const int nv = model.nv;

    VectorXd q_ref = node->q();  // reference = current config (zero)
    auto cost = std::make_shared<ConfigurationCost>(q_ref);
    node->add_cost(cost);

    VectorXd residual(nv);
    cost->evaluate(residual);

    EXPECT_LT(residual.norm(), 1e-12);
}

TEST_F(ConfigurationCostTest, NonZeroResidualAwayFromReference) {
    const int nv = model.nv;

    VectorXd q_ref = node->q();  // reference at zero
    node->q()(0) = 1.0;          // perturb current config

    auto cost = std::make_shared<ConfigurationCost>(q_ref);
    node->add_cost(cost);

    VectorXd residual(nv);
    cost->evaluate(residual);

    EXPECT_GT(residual.norm(), 1e-6);
}

TEST_F(ConfigurationCostTest, JacobianMatchesFiniteDifferences) {
    const int nv = model.nv;

    // Non-trivial state
    node->q()(0) = 0.7;
    node->v()(0) = 0.3;

    VectorXd q_ref(model.nq);
    q_ref(0) = 0.2;

    auto cost = std::make_shared<ConfigurationCost>(q_ref);
    node->add_cost(cost);

    // Analytical Jacobian (nv × 2*nv)
    MatrixXd analytical_jac(nv, 2 * nv);
    cost->jacobian(analytical_jac);

    // Numerical Jacobian via Lie-group perturbations (state only, no control)
    auto eval_func = [&]() -> VectorXd {
        VectorXd out(nv);
        cost->evaluate(out);
        return out;
    };

    MatrixXd numerical_jac = numerical_jacobian_node(
        eval_func, *node, nv, /*perturb_x=*/true, /*perturb_u=*/false);

    EXPECT_TRUE(jacobians_match(analytical_jac, numerical_jac, 1e-5, 1e-8));
}

TEST_F(ConfigurationCostTest, NoControlDependency) {
    const int nv = model.nv;

    VectorXd q_ref = node->q();
    auto cost = std::make_shared<ConfigurationCost>(q_ref);
    node->add_cost(cost);

    cost->jacobian();
    MatrixXd Ju = cost->get_jac_u();

    EXPECT_EQ(Ju.cols(), 0);
}
