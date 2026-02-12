#include "trajopt/ocp.h"
#include "trajopt/node.h"
#include "trajopt/constraints/integration_schemes/euler.h"
#include "sqp_solver/sqp_solver.h"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/joint/joint-generic.hpp>
#include <memory>

#include <iostream>

int main() {

    int N = 10;
    double dt = 0.05;

    OCP ocp(N);

    // Create a simple 1-DOF model (double integrator)
    pinocchio::Model robot_mdl;
    robot_mdl.name = "double_integrator";
    auto joint_id = robot_mdl.addJoint(
        0,
        pinocchio::JointModelPX(),
        pinocchio::SE3::Identity(),
        "slider"
    );
    robot_mdl.addJointFrame(joint_id);
    robot_mdl.lowerPositionLimit.resize(1);
    robot_mdl.upperPositionLimit.resize(1);
    robot_mdl.lowerPositionLimit << -10.0;
    robot_mdl.upperPositionLimit << 10.0;

    for (int i = 0; i < N; i++)
    {
        Node node(robot_mdl);
        node.add_dynamics(std::make_shared<EulerIntegration>(dt));
        ocp.addNode(std::move(node));
    }

    // Finalize: allocate trajectories and bind nodes
    ocp.finalize();

    // Set initial guess
    for (int k = 0; k < N; k++) {
        ocp.get_node(k).x().setZero();  // [q, v] = 0
        ocp.get_node(k).u().setOnes();  // control = 0
    }

    SQPSolver solver(ocp);
    solver.solve();

    std::cout<<"state_trajectory"<<std::endl;
    for (int k = 0; k < N; k++) 
        std::cout<<"x["<<k<<"]"<<ocp.get_node(k).x().transpose()<<std::endl;  
    
    std::cout<<"control_trajectory"<<std::endl;
    for (int k = 0; k < N; k++) 
        std::cout<<"u["<<k<<"]"<<ocp.get_node(k).u().transpose()<<std::endl;

    return 0;
}



