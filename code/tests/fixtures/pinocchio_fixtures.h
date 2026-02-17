#pragma once

#include <gtest/gtest.h>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/joint/joint-generic.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <trajopt/node.h>
#include <memory>
#include <random>

/**
 * Build a random robot: floating base + 2 revolute joints.
 *   nq = 9  (7 floating + 2 revolute)
 *   nv = 8  (6 floating + 2 revolute)
 */
inline pinocchio::Model buildRandomFloating2R(std::mt19937& gen) {
    std::uniform_real_distribution<double> mass_dist(0.5, 5.0);
    std::uniform_real_distribution<double> length_dist(0.2, 1.5);

    pinocchio::Model model;
    model.name = "random_floating_2r";

    // Floating base (6 DOF)
    double base_mass = mass_dist(gen);
    double base_size = length_dist(gen);
    pinocchio::Inertia base_inertia =
        pinocchio::Inertia::FromBox(base_mass, base_size, base_size, base_size);

    auto base_id = model.addJoint(
        0,
        pinocchio::JointModelFreeFlyer(),
        pinocchio::SE3::Identity(),
        "floating_base"
    );
    model.appendBodyToJoint(base_id, base_inertia);
    model.addJointFrame(base_id);

    // Revolute joint 1 (z-axis)
    double link1_mass = mass_dist(gen);
    double link1_length = length_dist(gen);
    pinocchio::Inertia link1_inertia =
        pinocchio::Inertia::FromCylinder(link1_mass, 0.05, link1_length);

    pinocchio::SE3 placement1 = pinocchio::SE3::Identity();
    placement1.translation() = Eigen::Vector3d(0., 0., base_size / 2.0);

    auto joint1_id = model.addJoint(
        base_id,
        pinocchio::JointModelRZ(),
        placement1,
        "revolute_1"
    );
    model.appendBodyToJoint(joint1_id, link1_inertia,
        pinocchio::SE3(Eigen::Matrix3d::Identity(),
                       Eigen::Vector3d(0., 0., link1_length / 2.0)));
    model.addJointFrame(joint1_id);

    // Revolute joint 2 (z-axis)
    double link2_mass = mass_dist(gen);
    double link2_length = length_dist(gen);
    pinocchio::Inertia link2_inertia =
        pinocchio::Inertia::FromCylinder(link2_mass, 0.05, link2_length);

    pinocchio::SE3 placement2 = pinocchio::SE3::Identity();
    placement2.translation() = Eigen::Vector3d(0., 0., link1_length);

    auto joint2_id = model.addJoint(
        joint1_id,
        pinocchio::JointModelRZ(),
        placement2,
        "revolute_2"
    );
    model.appendBodyToJoint(joint2_id, link2_inertia,
        pinocchio::SE3(Eigen::Matrix3d::Identity(),
                       Eigen::Vector3d(0., 0., link2_length / 2.0)));
    model.addJointFrame(joint2_id);

    // Position limits (needed for pinocchio::randomConfiguration)
    model.lowerPositionLimit.head<3>().setConstant(-10.0);
    model.upperPositionLimit.head<3>().setConstant(10.0);
    model.lowerPositionLimit.tail<2>().setConstant(-2.0 * M_PI);
    model.upperPositionLimit.tail<2>().setConstant(2.0 * M_PI);

    return model;
}

/**
 * Fixture providing a random floating-base + 2-revolute robot.
 *   nq = 9, nv = 8
 * Two nodes are set up (node + next_node) with trajectory storage bound.
 */
class DoubleIntegratorFixture : public ::testing::Test {
   protected:
    pinocchio::Model model;
    std::shared_ptr<Node> node;
    std::shared_ptr<Node> next_node;
    double dt = 0.1;

    // Trajectory storage for nodes
    VectorXd x_storage, u_storage;
    VectorXd x_next_storage, u_next_storage;

    void SetUp() override {
        std::mt19937 gen(std::random_device{}());
        model = buildRandomFloating2R(gen);

        node = std::make_shared<Node>(model);
        next_node = std::make_shared<Node>(model);
        node->next_node = next_node.get();

        // Allocate trajectory storage
        x_storage.setZero(node->nx());
        u_storage.setZero(node->nu());
        x_next_storage.setZero(next_node->nx());
        u_next_storage.setZero(next_node->nu());

        // Initialize configurations to a valid point on the manifold
        // (identity quaternion for floating base)
        VectorXd q0 = pinocchio::neutral(model);
        x_storage.head(model.nq) = q0;
        x_next_storage.head(model.nq) = q0;

        // Bind nodes to storage
        node->bind_trajectory(&x_storage, &u_storage);
        next_node->bind_trajectory(&x_next_storage, &u_next_storage);
    }

    void TearDown() override {
        node.reset();
        next_node.reset();
    }
};
