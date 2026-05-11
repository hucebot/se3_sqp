import sys
from pathlib import Path
import numpy as np
import time
import threading
from scipy.spatial.transform import Rotation as R
from utils import plotter, interactive_marker, rviser


BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
sys.path.insert(0, str(BUILD_DIR))

import sqp_solver as sqp
import pinocchio as pin

RESOURCES = Path(__file__).resolve().parent.parent / "resources"

def main():
    T = 0.1
    N  = 10
    dt = T/N
    print(f"dt: {dt}")

    urdf_path = str(RESOURCES / "urdf/floating_frame.urdf")
    print(f"URDF: {urdf_path}")
    rviz = rviser.rviser(urdf_path)

    ocp = sqp.OCP(N)

    # Initial configuration
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
    q0 = pin.neutral(model)
    print(f"q0: {q0}")

    data = pin.Data(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    # Running nodes
    v_lims = np.array([10., 10., 10., 10., 10., 10.])
    for k in range(N - 1):
        model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
        node = sqp.Node(model)

        node.add_dynamics(sqp.SemiEulerIntegration(dt))

        if k > 0:
            node.add_cost(sqp.FrameVelocityCost("base_link", np.array([0., 0., 0., 0., 0., 0.]), 1e-3))
            node.add_constraint(sqp.FrameVelocityConstraint("base_link", v_lims))

        node.add_cost(sqp.FrameAccelerationCost("base_link", np.array([0., 0., 0., 0., 0., 0.]), 1e-3))

        ocp.addNode(node)


    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
    node = sqp.Node(model)

    translation_task = sqp.FrameTranslationCost("base_link", q0[0:3], 1e3)
    orientation_task = sqp.FrameOrientationCost("base_link", R.from_quat(q0[3:]).as_matrix(), 1e3) # xyzw

    node.add_cost(translation_task)
    node.add_cost(orientation_task)
    node.add_cost(sqp.FrameVelocityCost("base_link", np.array([0., 0., 0., 0., 0., 0.]), 1e6))
    node.add_constraint(sqp.FrameVelocityConstraint("base_link", v_lims))
    ocp.addNode(node)

    ocp.finalize()

    # Initial guess: downward rest position, zero velocity and control
    for k in range(N-1):
        ocp.get_node(k).q()[:] = q0
        ocp.get_node(k).v()[:] = 0.0
        ocp.get_node(k).u()[:] = 0.0
    ocp.get_node(N-1).q()[:] = q0

        # Solve
    solver = sqp.SQPSolver(ocp)
    opts = sqp.SQPoptions()
    opts.max_sqp_iters = 100
    opts.hpipm_iter_max = 20
    opts.eps_inequality = 1e-4
    opts.ls_type       = sqp.LSType.MERIT
    opts.ls_merit_mu = 10.
    opts.ls_merit_eta = 1e-4
    solver.set_options(opts)

    solver.solve()

    opts = sqp.SQPoptions()
    opts.max_sqp_iters = 1
    opts.verbose = 0
    opts.ls_type       = sqp.LSType.NONE
    solver.set_options(opts)

    # Markers
    lock = threading.Lock()
    interactive_marker.interactive_marker(rviz.server, translation_task, orientation_task, lock)

    # Plot
    #solver_plot = plotter.plot_solver_stats(rviz.server, dt)
    v_plot = plotter.plot(title="Velocities", size=model.nv, legend_label="v", server=rviz.server, dt=dt)

    while True:
        #ocp.x_traj()[0][:] = ocp.x_traj()[1][:]
        solver.solve(ocp.x_traj()[0][:])

        q1 = ocp.x_traj()[1][0:model.nq]
        v1 = ocp.x_traj()[1][model.nq:]

        rviz.update(q_base=q1)

        v_plot.update(v1)

        for k in range(N-2):
            ocp.get_node(k).q()[:] = ocp.get_node(k+1).q()[:]
            ocp.get_node(k).v()[:] = ocp.get_node(k+1).v()[:]
            ocp.get_node(k).u()[:] = ocp.get_node(k+1).u()[:]

        #solver_plot.update(solver.get_stats())
        time.sleep(dt)


if __name__ == "__main__":
    main()
