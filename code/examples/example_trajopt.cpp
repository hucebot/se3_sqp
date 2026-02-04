#include "trajopt/ocp.h"
#include "trajopt/node.h"
#include "trajopt/constraints/integration_schemes/euler.h"
// #include "sqp_solver/sqp_solver.h"

#include <memory>

int main() {

    int N = 20;
    double dt = 0.01;

    OCP ocp;

    pinocchio::Model robot_mdl;

    for (int i = 0; i < N; i++)
    {
        Node node(robot_mdl);
        node.add_constraint(std::make_shared<EulerIntegration>(dt));
        ocp.addNode(node);
    }

    // Set initial guess
    for (int k = 0; k < N; k++) {
        ocp.get_node(k).x().setZero();  // [q, v] = 0
        ocp.get_node(k).u().setZero();  // control = 0
    }

    // SQPSolver solver;
    // solver.solve(ocp);

    return 0;
}



