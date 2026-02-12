#pragma once

#include <gtest/gtest.h>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
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

    // Trajectory storage for nodes
    VectorXd x_storage, u_storage;
    VectorXd x_next_storage, u_next_storage;

    void SetUp() override {
        // model.name = "double_integrator";
        // auto joint_id = model.addJoint(
        //     0,
        //     pinocchio::JointModelPX(),
        //     pinocchio::SE3::Identity(),
        //     "slider"
        // );
        // model.addJointFrame(joint_id);
        // model.lowerPositionLimit.resize(1);
        // model.upperPositionLimit.resize(1);
        // model.lowerPositionLimit << -10.0;
        // model.upperPositionLimit << 10.0;

        const std::string urdf_path = "/workspace/code/resources/pendubot.urdf";
        pinocchio::urdf::buildModel(urdf_path, model);

        node = std::make_shared<Node>(model);
        next_node = std::make_shared<Node>(model);
        node->next_node = next_node.get();

        // Allocate trajectory storage
        x_storage.setZero(node->nx());
        u_storage.setZero(node->nu());
        x_next_storage.setZero(next_node->nx());
        u_next_storage.setZero(next_node->nu());

        // Bind nodes to storage
        node->bind_trajectory(&x_storage, &u_storage);
        next_node->bind_trajectory(&x_next_storage, &u_next_storage);
    }

    void TearDown() override {
        node.reset();
        next_node.reset();
    }
};
