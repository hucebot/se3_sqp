#pragma once

#include <gtest/gtest.h>
#include <pinocchio/multibody/model.hpp>
#include <trajopt/node.h>
#include <memory>

/**
 * Fixture providing a simple 1-DOF prismatic joint model.
 * Equivalent to a double integrator: x'' = u
 */
class DoubleIntegratorFixture : public ::testing::Test {
   protected:
    pinocchio::Model model;
    std::shared_ptr<Node> node;
    std::shared_ptr<Node> next_node;
    double dt = 0.1;

    void SetUp() override {
        model.name = "double_integrator";
        auto joint_id = model.addJoint(
            0,
            pinocchio::JointModelPX(),
            pinocchio::SE3::Identity(),
            "slider"
        );
        model.addJointFrame(joint_id);
        model.lowerPositionLimit.resize(1);
        model.upperPositionLimit.resize(1);
        model.lowerPositionLimit << -10.0;
        model.upperPositionLimit << 10.0;

        node = std::make_shared<Node>(model);
        next_node = std::make_shared<Node>(model);
        node->next_node = next_node.get();
    }

    void TearDown() override {
        node.reset();
        next_node.reset();
    }
};
