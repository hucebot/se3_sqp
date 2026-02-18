#include <gtest/gtest.h>
#include <trajopt/functions/frame_translation.h>
#include <trajopt/node.h>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include "pinocchio_fixtures.h"
#include "numerical_differentiation.h"

class FrameTranslationTest : public DoubleIntegratorFixture {};

TEST_F(FrameTranslationTest, ZeroResidualAtCurrentPosition) {
    node->q() = pinocchio::randomConfiguration(model);

    // Compute current frame position
    pinocchio::forwardKinematics(model, node->data(), node->q());
    pinocchio::updateFramePlacements(model, node->data());
    int frame_id = model.getFrameId("end_effector");
    Eigen::Vector3d p_current = node->data().oMf[frame_id].translation();

    FrameTranslation ft("end_effector", p_current);
    ft.set_node(node.get());
    ft.allocate_slices();

    ft.evaluate();

    EXPECT_LT(ft.get_value().norm(), 1e-12);
}

TEST_F(FrameTranslationTest, NonZeroResidualAwayFromReference) {
    node->q() = pinocchio::randomConfiguration(model);
    Eigen::Vector3d p_ref(100.0, 100.0, 100.0);

    FrameTranslation ft("end_effector", p_ref);
    ft.set_node(node.get());
    ft.allocate_slices();

    ft.evaluate();

    EXPECT_GT(ft.get_value().norm(), 1e-6);
}

TEST_F(FrameTranslationTest, OutputDimIs3) {
    FrameTranslation ft("end_effector");
    ft.set_node(node.get());
    ft.allocate_slices();

    EXPECT_EQ(ft.get_output_dim(), 3);
}

TEST_F(FrameTranslationTest, JacobianMatchesFiniteDifferences) {
    const int nv = model.nv;

    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);

    Eigen::Vector3d p_ref = Eigen::Vector3d::Random();

    FrameTranslation ft("end_effector", p_ref);
    ft.set_node(node.get());
    ft.allocate_slices();

    ft.jacobian();
    MatrixXd analytical_jac = ft.get_jac_x();

    auto eval_func = [&]() -> VectorXd {
        ft.evaluate();
        return ft.get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian_node(
        eval_func, *node, 3, /*perturb_x=*/true, /*perturb_u=*/false);

    EXPECT_TRUE(jacobians_match(analytical_jac, numerical_jac, 1e-5, 1e-8));
}

TEST_F(FrameTranslationTest, JacobianMatchesFiniteDifferencesContact1) {
    const int nv = model.nv;

    node->q() = pinocchio::randomConfiguration(model);
    node->v() = VectorXd::Random(nv);

    Eigen::Vector3d p_ref = Eigen::Vector3d::Random();

    FrameTranslation ft("contact_1", p_ref);
    ft.set_node(node.get());
    ft.allocate_slices();

    ft.jacobian();
    MatrixXd analytical_jac = ft.get_jac_x();

    auto eval_func = [&]() -> VectorXd {
        ft.evaluate();
        return ft.get_value();
    };

    MatrixXd numerical_jac = numerical_jacobian_node(
        eval_func, *node, 3, /*perturb_x=*/true, /*perturb_u=*/false);

    EXPECT_TRUE(jacobians_match(analytical_jac, numerical_jac, 1e-5, 1e-8));
}

TEST_F(FrameTranslationTest, NoControlDependency) {
    FrameTranslation ft("end_effector");
    ft.set_node(node.get());
    ft.allocate_slices();

    ft.jacobian();
    MatrixXd Ju = ft.get_jac_u();

    EXPECT_TRUE(Ju.isZero());
}

TEST_F(FrameTranslationTest, InvalidFrameThrows) {
    FrameTranslation ft("nonexistent_frame");
    ft.set_node(node.get());

    EXPECT_THROW(ft.allocate_slices(), std::invalid_argument);
}
