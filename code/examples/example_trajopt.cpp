#include "trajopt/ocp.h"
#include "trajopt/node.h"
#include "sqp_solver/sqp_solver.h"

int main() {

    int N = 20;
    int dt = 0.01;
    
    OCP ocp;

    for (int i = 0; i < N; i++)
    {
        Node node;
        

        ocp.addNode(node);
    }


    SQPSolver solver;

    solver.solve();

    
    
 }



