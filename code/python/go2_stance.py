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

    feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    # Initial configuration
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
    q0 = pin.neutral(model)
    q0[2] = 0.30
    q0[7:] = np.array([0., 0.8, -1.6, 0., 0.8, -1.6, 0., 0.8, -1.6, 0., 0.8, -1.6])


    data = pin.Data(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    # Find lowest foot z-coordinate
    min_foot_z = 1e10
    for foot in feet:
        fid = model.getFrameId(foot)
        fz = data.oMf[fid].translation[2]
        min_foot_z = min(min_foot_z, fz)
    # Adjust base height so feet are at ground level (z=0)
    q0[2] -= min_foot_z

    # Running nodes
    for k in range(N - 1):
        model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
        node = sqp.Node(model)

        for foot in feet:
            node.add_contact(foot)

        node.add_dynamics(sqp.SemiEulerIntegration(dt))
        node.add_constraint(sqp.InvDynamics())

        for foot in feet:
            node.add_constraint(sqp.ContactConstraint(foot))
            node.add_constraint(sqp.FrictionConeConstraint(foot, 0.8))

        node.add_cost(sqp.ConfigurationCost(q0, 1e-3))
        node.add_cost(sqp.VelocityCost(1e-3))
        node.add_cost(sqp.AccelerationCost(1e-9))
        ocp.addNode(node)

    # Terminal node — high weight on reaching the target
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
    node = sqp.Node(model)
    node.add_cost(sqp.ConfigurationCost(q0, 1e-3))
    node.add_cost(sqp.VelocityCost(1e3))
    ocp.addNode(node)

    ocp.finalize()

    # Initial guess: downward rest position, zero velocity and control
    print("q0:", q0)
    for k in range(N):
        ocp.get_node(k).q()[:] = q0
        ocp.get_node(k).v()[:] = 0.0
        ocp.get_node(k).u()[:] = 0.0


    # Solve
    solver = sqp.SQPSolver(ocp)
    opts = sqp.SQPoptions()
    opts.max_sqp_iters = 200
    opts.tolerance     = 1e-3
    #opts.ls_merit_eta  = 1e-4
    opts.ls_type       = sqp.LSType.MERIT
    solver.set_options(opts)

    solver.solve()

    ocp.save_trajectory("/workspace/code/resources/trajectories/go2_stance.json",
                        dt, urdf_path)


if __name__ == "__main__":
    main()
