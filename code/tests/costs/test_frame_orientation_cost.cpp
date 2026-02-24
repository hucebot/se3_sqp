#include <gtest/gtest.h>
#include <trajopt/costs/frame_orientation_cost.h>
#include <trajopt/node.h>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include "pinocchio_fixtures.h"
#include "numerical_differentiation.h"

class FrameOrientationCostTest : public DoubleIntegratorFixture {};

TEST_F(FrameOrientationCostTest, ZeroResidualAtCurrentOrientation) {
    node->q() = pinocchio::randomConfiguration(model);

    pinocchio::forwardKinematics(model, node->data(), node->q());
    pinocchio::updateFramePlacements(model, node->data());
    int frame_id = model.getFrameId("end_effector");
    Eigen::Matrix3d R_current = node->data().oMf[frame_id].rotation();

    auto cost = std::make_shared<FrameOrientationCost>("end_effector", R_current);
    node->add_cost(cost);

    cost->evaluate();

    EXPECT_LT(cost->get_value().norm(), 1e-12);
}

TEST_F(FrameOrientationCostTest, NonZeroResidualAwayFromReference) {
    node->q() = pinocchio::randomConfiguration(model);

    // A fixed reference that is unlikely to match the random config
    Eigen::Matrix3d R_ref;
    R_ref = Eigen::AngleAxisd(1.0, Eigen::Vector3d::UnitX()).toRotationMatrix();

    auto cost = std::make_shared<FrameOrientationCost>("end_effector", R_ref);
    node->add_cost(cost);

    cost->evaluate();

    EXPECT_GT(cost->get_value().norm(), 1e-6);
}

TEST_F(FrameOrientationCostTest, OutputDimIs3) {
    auto cost = std::make_shared<FrameOrientationCost>("end_effector");
    node->add_cost(cost);

    EXPECT_EQ(cost->get_output_dim(), 3);
}

TEST_F(FrameOrientationCostTest, JacobianMatchesFiniteDifferences) {
    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(model.nv);

    Eigen::Matrix3d R_ref =
        Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    auto cost = std::make_shared<FrameOrientationCost>("end_effector", R_ref);
    node->add_cost(cost);

    cost->jacobian();
    MatrixXd analytical_jac = cost->get_jac_x();

    auto eval_func = [&]() -> VectorXd {
        cost->evaluate();
        return cost->get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian_node(
        eval_func, *node, 3, /*perturb_x=*/true, /*perturb_u=*/false);

    EXPECT_TRUE(jacobians_match(analytical_jac, numerical_jac, 1e-5, 1e-8));
}

TEST_F(FrameOrientationCostTest, NoControlDependency) {
    auto cost = std::make_shared<FrameOrientationCost>("end_effector");
    node->add_cost(cost);

    cost->jacobian();
    MatrixXd Ju = cost->get_jac_u();

    EXPECT_TRUE(Ju.isZero());
}

TEST_F(FrameOrientationCostTest, InvalidFrameThrows) {
    auto cost = std::make_shared<FrameOrientationCost>("nonexistent_frame");
    EXPECT_THROW(node->add_cost(cost), std::invalid_argument);
}
