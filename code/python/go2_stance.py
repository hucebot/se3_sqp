import sys
import math
from pathlib import Path

import numpy as np

BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
sys.path.insert(0, str(BUILD_DIR))

import sqp_solver as sqp
import pinocchio as pin

RESOURCES = Path(__file__).resolve().parent.parent / "resources"

def main():
    N  = 100
    dt = 0.01

    urdf_path = str(RESOURCES / "urdf/go2/go2.urdf")
    print(f"URDF: {urdf_path}")

    ocp = sqp.OCP(N)

    feet = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]

    # Initial configuration
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
    q_init = pin.neutral(model)
    print("Neutral configuration:", q_init)
    q_init[:7] = np.array([0., 0., 0.296797, 0., 0., 0., 1.])
    q_init[7:] = np.array([0., 0.8, -1.6, 0., 0.8, -1.6, 0., 0.8, -1.6, 0., 0.8, -1.6])
    print("qinit:", q_init)
    q_ref = q_init.copy()  # target configuration is the same as initial

    # Running nodes
    for _ in range(N - 1):
        model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
        node = sqp.Node(model)

        for foot in feet:
            node.add_contact(foot)

        node.add_dynamics(sqp.SemiEulerIntegration(dt))
        node.add_constraint(sqp.InvDynamics())

        for foot in feet:
            node.add_constraint(sqp.ContactConstraint(foot))
            node.add_constraint(sqp.FrictionConeConstraint(foot, 0.8))

        node.add_cost(sqp.ConfigurationCost(q_ref, 1e-3))
        node.add_cost(sqp.VelocityCost(1e-6))
        node.add_cost(sqp.AccelerationCost(1e-9))
        ocp.addNode(node)

    # Terminal node — high weight on reaching the target
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
    node = sqp.Node(model)
    # node.add_cost(sqp.ConfigurationCost(q_ref, 1e0))
    node.add_cost(sqp.VelocityCost(1e0))
    ocp.addNode(node)

    ocp.finalize()

    # Initial guess: downward rest position, zero velocity and control
    for k in range(N):
        ocp.get_node(k).q()[:] = q_ref
        ocp.get_node(k).v()[:] = 0.0
        ocp.get_node(k).u()[:] = 0.0


    # Solve
    solver = sqp.SQPSolver(ocp)
    opts = sqp.SQPoptions()
    opts.max_sqp_iters = 1000
    opts.tolerance     = 1e-3
    opts.ls_merit_eta  = 1e-4
    opts.ls_type       = sqp.LSType.MERIT
    solver.set_options(opts)

    solver.solve()

    ocp.save_trajectory("/workspace/code/resources/trajectories/go2_stance.json",
                        dt, urdf_path)


if __name__ == "__main__":
    main()
