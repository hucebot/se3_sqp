#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <pinocchio/multibody/joint/joint-generic.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "sqp_solver/sqp_solver.h"
#include "trajopt/constraints/integration_schemes/euler.h"
#include "trajopt/constraints/integration_schemes/semi-euler.h"
#include "trajopt/constraints/inverse_dynamics.h"
#include "trajopt/costs/configuration_cost.h"
#include "trajopt/costs/velocity_cost.h"
#include "trajopt/costs/acceleration_cost.h"
#include "trajopt/node.h"
#include "trajopt/ocp.h"

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    int N = 100;
    double dt = 0.01;

    OCP ocp(N);

    pinocchio::Model robot_mdl;
    const std::string urdf_path = "/workspace/code/resources/pendubot.urdf";
    pinocchio::urdf::buildModel(urdf_path, robot_mdl);

    Vector2d q_ref;
    q_ref << M_PI, 0.;

    Vector2d v_ref;
    v_ref << 0., 0.;

    //TODO tip cost

    // Running nodes
    for (int i = 0; i < N - 1; i++) {
        Node node(robot_mdl);

        node.add_dynamics(std::make_shared<SemiEulerIntegration>(dt));
        node.add_constraint(std::make_shared<InvDynamics>());        

        node.add_cost(std::make_shared<ConfigurationCost>(q_ref, 0.));
        node.add_cost(std::make_shared<VelocityCost>(1e-6));
        node.add_cost(std::make_shared<AccelerationCost>(1e-9));

        ocp.addNode(std::move(node));
    }

    // Terminal node
    {
        Node node(robot_mdl);

        node.add_cost(std::make_shared<ConfigurationCost>(q_ref, 1e0));
        node.add_cost(std::make_shared<VelocityCost>(1e0));

        ocp.addNode(std::move(node));
    }

    // Finalize: allocate trajectories and bind nodes
    ocp.finalize();

    // Set initial guess
    for (int k = 0; k < N; k++) {
        ocp.get_node(k).q() << 0.0, 0.;
        ocp.get_node(k).v().setZero();
        ocp.get_node(k).u().setZero();  // control = 0
    }

    SQPSolver solver(ocp);

    SQPoptions opts;
    opts.max_sqp_iters = 1000;
    opts.tolerance = 1e-3;
    opts.ls_merit_eta = 1e-4;
    opts.ls_type = LSType::MERIT;
    solver.set_options(opts);

    solver.solve();

    ocp.save_trajectory("/workspace/code/resources/trajectories/trajectory.json", dt, urdf_path);


    return 0;
}
