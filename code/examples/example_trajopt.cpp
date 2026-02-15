#include "trajopt/ocp.h"
#include "trajopt/node.h"
#include "sqp_solver/sqp_solver.h"

#include "trajopt/constraints/integration_schemes/euler.h"
#include "trajopt/constraints/inverse_dynamics.h"
#include "trajopt/costs/configuration_cost.h"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/joint/joint-generic.hpp>
#include <memory>

#include <iostream>
#include <cstdlib>
#include <ctime>

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    int N = 300;
    double dt = 0.01;

    OCP ocp(N);

    pinocchio::Model robot_mdl;
    const std::string urdf_path = "/workspace/code/resources/pendubot.urdf";
    pinocchio::urdf::buildModel(urdf_path, robot_mdl);

    Vector2d q_ref;
    q_ref << M_PI, 0.;

    for (int i = 0; i < N; i++)
    {
        Node node(robot_mdl);
        if (i<N-1)
        {
            node.add_dynamics(std::make_shared<EulerIntegration>(dt));
            node.add_constraint(std::make_shared<InvDynamics>());
        }

        if (i==N-1) node.add_cost(std::make_shared<ConfigurationCost>(q_ref));
            
        ocp.addNode(std::move(node));
    }

    // Finalize: allocate trajectories and bind nodes
    ocp.finalize();

    // Set initial guess
    for (int k = 0; k < N; k++) {
        ocp.get_node(k).q() << 0.1, 0.0;
        ocp.get_node(k).v().setZero();
        ocp.get_node(k).u().setZero();             // control = 0
    }

    SQPSolver solver(ocp);
    solver.solve();

    ocp.save_trajectory("/workspace/trajectory.json", dt, urdf_path);
    std::cout << "Trajectory saved to /workspace/trajectory.json" << std::endl;

    return 0;
}



