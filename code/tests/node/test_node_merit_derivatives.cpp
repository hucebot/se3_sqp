#include <gtest/gtest.h>
#include <trajopt/constraints/inverse_dynamics.h>
#include <trajopt/costs/configuration_cost.h>
#include <trajopt/node.h>
#include "pinocchio_fixtures.h"
#include "numerical_differentiation.h"

class NodeMeritDerivativeTest : public DoubleIntegratorFixture {};

/**
 * Central-difference approximation of d/dα φ(x⊕αdx, u+αdu)|_{α=0}.
 *
 * compute_phi() must evaluate φ from the node's current (x, u).
 * (x, u) are restored to (x0, u0) before returning.
 */
static double fd_directional(
    Node& n,
    std::function<double()> compute_phi,
    VectorXdConstRef dx,
    VectorXdConstRef du,
    double eps = 1e-6)
{
    VectorXd x0 = n.x();
    VectorXd u0 = n.u();
    VectorXd x_pert(n.nx());

    VectorXd dx_pos = eps * dx;
    n.x_oplus(x0, dx_pos, x_pert);
    n.x() = x_pert;
    n.u() = u0 + eps * du;
    const double phi_plus = compute_phi();

    VectorXd dx_neg = -eps * dx;
    n.x_oplus(x0, dx_neg, x_pert);
    n.x() = x_pert;
    n.u() = u0 - eps * du;
    const double phi_minus = compute_phi();

    n.x() = x0;
    n.u() = u0;

    return (phi_plus - phi_minus) / (2.0 * eps);
}

// =============================================================================
// Cost gradient
// =============================================================================

TEST_F(NodeMeritDerivativeTest, CostGradientMatchesFiniteDifferences) {
    // Random state and control
    node->q() = VectorXd::Random(model.nq);
    node->v() = VectorXd::Random(model.nv);
    node->u() = VectorXd::Random(node->nu());

    VectorXd q_ref = VectorXd::Random(model.nq);
    auto cost = std::make_shared<ConfigurationCost>(q_ref);
    node->add_cost(cost);

    // Compute gradient once at the current (x, u)
    cost->evaluate();
    cost->jacobian();
    node->calc_cost_gradient();

    const VectorXd grad_x = node->get_cost_grad_x();
    const VectorXd grad_u = node->get_cost_grad_u();

    // Cost is smooth — FD should match to high accuracy over many random directions
    for (int trial = 0; trial < 10; ++trial) {
        const VectorXd dx = VectorXd::Random(node->ndx());
        const VectorXd du = VectorXd::Random(node->ndu());

        const double analytical = grad_x.dot(dx) + grad_u.dot(du);

        auto compute_phi = [&]() -> double {
            cost->evaluate();
            node->calc_cost();
            return node->get_cost();
        };
        const double numerical = fd_directional(*node, compute_phi, dx, du);

        EXPECT_NEAR(analytical, numerical,
                    1e-4 * std::max(1.0, std::abs(numerical)))
            << "trial=" << trial;
    }
}

// =============================================================================
// Defect gradient
// =============================================================================

TEST_F(NodeMeritDerivativeTest, DefectGradientMatchesFiniteDifferences) {
    // Run several random states; for each one verify that the argmax component
    // is well-separated so that the L∞ norm is smooth at that point.
    auto dyn = std::make_shared<InvDynamics>();
    node->add_dynamics(dyn);

    int valid_trials = 0;
    for (int seed = 0; seed < 20 && valid_trials < 5; ++seed) {
        node->q() = VectorXd::Random(model.nq);
        node->v() = VectorXd::Random(model.nv);
        node->u() = VectorXd::Random(node->nu());

        dyn->evaluate();
        dyn->jacobian();
        node->calc_dynamics_defect();
        node->calc_defect_gradient();

        const VectorXd& g = dyn->get_value();
        const VectorXd abs_g = g.cwiseAbs();

        // Require a nonzero defect
        if (node->get_dynamics_defect() < 1e-6) continue;

        // Require the argmax to be unique (gap ≥ 10 % of max) so FD is reliable
        int k_star;
        const double max_g = abs_g.maxCoeff(&k_star);
        VectorXd tmp = abs_g;
        tmp(k_star) = 0.0;
        const double second_max = tmp.maxCoeff();
        if (max_g - second_max < 0.1 * max_g) continue;

        const VectorXd grad_x = node->get_defect_grad_x();
        const VectorXd grad_u = node->get_defect_grad_u();

        // Use small random directions so the perturbation stays on the same branch
        for (int t = 0; t < 3; ++t) {
            const VectorXd dx = 0.01 * VectorXd::Random(node->ndx());
            const VectorXd du = 0.01 * VectorXd::Random(node->ndu());

            const double analytical = grad_x.dot(dx) + grad_u.dot(du);

            auto compute_phi = [&]() -> double {
                dyn->evaluate();
                node->calc_dynamics_defect();
                return node->get_dynamics_defect();
            };
            const double numerical = fd_directional(*node, compute_phi, dx, du);

            EXPECT_NEAR(analytical, numerical,
                        1e-4 * std::max(1.0, std::abs(numerical)))
                << "seed=" << seed << " t=" << t;
        }
        ++valid_trials;
    }

    EXPECT_GE(valid_trials, 5)
        << "not enough valid random states found — check dynamics residuals";
}

// =============================================================================
// Violation gradient
// =============================================================================

TEST_F(NodeMeritDerivativeTest, ViolationGradientMatchesFiniteDifferences) {
    auto con = std::make_shared<InvDynamics>();
    node->add_constraint(con);

    int valid_trials = 0;
    for (int seed = 0; seed < 20 && valid_trials < 5; ++seed) {
        node->q() = VectorXd::Random(model.nq);
        node->v() = VectorXd::Random(model.nv);
        node->u() = VectorXd::Random(node->nu());

        // Evaluate residual to design bounds with a clear dominant violation.
        con->evaluate();
        const VectorXd g0 = con->get_value();

        // Force component 0 to have the largest upper-bound violation.
        // Component 1 gets a smaller (but nonzero) violation.
        VectorXd lb = g0.array() - 1e6;   // no lower violations
        VectorXd ub = g0;
        ub(0) -= 2.0;   // component 0: violation = 2.0
        ub(1) -= 0.5;   // component 1: violation = 0.5
        con->set_bounds(lb, ub);

        con->jacobian();
        node->calc_constraint_violation();

        if (node->get_constraint_violation() < 1.5) continue;  // sanity check

        node->calc_violation_gradient();

        const VectorXd grad_x = node->get_violation_grad_x();
        const VectorXd grad_u = node->get_violation_grad_u();

        for (int t = 0; t < 3; ++t) {
            const VectorXd dx = 0.01 * VectorXd::Random(node->ndx());
            const VectorXd du = 0.01 * VectorXd::Random(node->ndu());

            const double analytical = grad_x.dot(dx) + grad_u.dot(du);

            auto compute_phi = [&]() -> double {
                con->evaluate();
                node->calc_constraint_violation();
                return node->get_constraint_violation();
            };
            const double numerical = fd_directional(*node, compute_phi, dx, du);

            EXPECT_NEAR(analytical, numerical,
                        1e-4 * std::max(1.0, std::abs(numerical)))
                << "seed=" << seed << " t=" << t;
        }
        ++valid_trials;
    }

    EXPECT_GE(valid_trials, 5)
        << "not enough valid random states found";
}
