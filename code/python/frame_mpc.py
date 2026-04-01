import sys
import math
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np
import viser
from yourdfpy import URDF
from viser.extras import ViserUrdf
import time


class rvizer:
    def __init__(self, URDF_PATH):
        self.server = viser.ViserServer(verbose=True)

        self.urdf = URDF.load(
                URDF_PATH,
                load_meshes=True,
                build_scene_graph=True,
                load_collision_meshes=True,
                build_collision_scene_graph=True,
            )

        self.base_frame = self.server.scene.add_frame("/robot_base", show_axes=False)
        self.viser_urdf = ViserUrdf(
                self.server,
                urdf_or_path=self.urdf,
                root_node_name="/robot_base",
                load_meshes=True
            )

        self.server.scene.add_grid(
                "/ground_grid",
                width=5.0,
                height=5.0,
                width_segments=20,
                height_segments=20,
            )

        self.server.initial_camera.position = (1.5, 1.5, 1.5)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.5)
        self.server.initial_camera.up = (0.0, 0.0, 1.0)


BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
sys.path.insert(0, str(BUILD_DIR))

import sqp_solver as sqp
import pinocchio as pin

RESOURCES = Path(__file__).resolve().parent.parent / "resources"

def main():
    N  = 20
    dt = 0.001

    urdf_path = str(RESOURCES / "urdf/floating_frame.urdf")
    print(f"URDF: {urdf_path}")


    ocp = sqp.OCP(N)

    # Initial configuration
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
    q0 = pin.neutral(model)
    print(f"q0: {q0}")

    data = pin.Data(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    # Running nodes
    p0 = sqp.FrameTranslationConstraint("base_link", q0[0:3])
    v0 = sqp.FrameVelocityConstraint("base_link", np.array([0., 0., 0., 0., 0., 0.]))
    for k in range(N - 1):
        model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
        node = sqp.Node(model)

        node.add_dynamics(sqp.SemiEulerIntegration(dt))

        if k == 0:
            node.add_constraint(p0)
            node.add_constraint(v0)

        node.add_cost(sqp.VelocityCost(1e-6))
        node.add_cost(sqp.AccelerationCost(1e-9))

        ocp.addNode(node)


    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
    node = sqp.Node(model)

    qf = q0.copy()
    qf[2] = 1.
    qf[6] = 0.
    qf[3] = 1.
    Rmat = R.from_quat(qf[3:]).as_matrix()

    translation_task = sqp.FrameTranslationCost("base_link", qf[0:3], 1e3)
    orientation_task = sqp.FrameOrientationCost("base_link", Rmat, 1e3)

    node.add_cost(translation_task)
    #node.add_cost(orientation_task)
    node.add_cost(sqp.VelocityCost(1e3))
    ocp.addNode(node)

    ocp.finalize()

    # Initial guess: downward rest position, zero velocity and control
    for k in range(N):
        ocp.get_node(k).q()[:] = q0
        ocp.get_node(k).v()[:] = 0.0
        ocp.get_node(k).u()[:] = 0.0

        # Solve
    solver = sqp.SQPSolver(ocp)
    opts = sqp.SQPoptions()
    opts.max_sqp_iters = 200
    opts.tolerance     = 1e-3
    opts.hpipm_tol_eq     = 1e-3
    opts.hpipm_tol_ineq     = 1e-3
    #opts.ls_merit_eta  = 1e-4
    opts.ls_type       = sqp.LSType.MERIT
    solver.set_options(opts)

    solver.solve()
    q1 = ocp.x_traj()[1][0:model.nq]
    v1 = ocp.x_traj()[1][model.nq:]
    p0.set_ref(q1[0:3])
    v0.set_ref(v1)

    opts = sqp.SQPoptions()
    opts.max_sqp_iters = 1
    solver.set_options(opts)

    rviz = rvizer(urdf_path)
    while True:
        solver.solve()
        q1 = ocp.x_traj()[1][0:model.nq]
        v1 = ocp.x_traj()[1][model.nq:]

        #q1 = ocp.x_traj()[k][0:model.nq]
        pos = np.array(q1[:3])
        quat_xyzw = q1[3:7]

        print(f"q1: {q1.T}")
        p0.set_ref(pos)
        v0.set_ref(v1)


        rviz.base_frame.position = pos
        rviz.base_frame.wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        time.sleep(dt)


    #ocp.save_trajectory("/workspace/code/resources/trajectories/frame.json", dt, urdf_path)

if __name__ == "__main__":
    main()
