import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np
import viser
from yourdfpy import URDF
from viser.extras import ViserUrdf
import time
import threading
from pynput import keyboard
import pinocchio

vx = 0.0
vy = 0.0
wz = 0.0

LIN_STEP = 0.1
ANG_STEP = 0.2

def on_press(key):
    global vx, vy, wz

    try:
        if key.char == '8':
            vx += LIN_STEP
        elif key.char == '2':
            vx += -LIN_STEP
        elif key.char == '4':
            wz += ANG_STEP
        elif key.char == '6':
            wz += -ANG_STEP
    except AttributeError:
        pass


def get_velocity():
    return vx, vy, wz


listener = keyboard.Listener(
    on_press=on_press,
)

listener.start()

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

        self.force_visualization_initied = False
        self.force_handles = {}

    def get_link_scene_paths(self) -> dict[str, str]:
        """Map link names to their viser scene paths by inspecting joint frames."""
        paths = {}
        for joint, frame_handle in zip(
            self.viser_urdf._joint_map_values, self.viser_urdf._joint_frames
        ):
            paths[joint.child] = frame_handle.name
        return paths

    def update_forces(self, contact_forces: dict, scale: float = 0.05) -> None:
            """Draw contact force arrows in the contact frames."""
            for frame_name, force_local in contact_forces.items():
                f_local = np.array(force_local, dtype=np.float32)
                f_norm = float(np.linalg.norm(f_local))

                # Line segment from origin to scaled force, in the contact frame
                tip = scale * f_local
                points = np.array([[[0, 0, 0], tip]], dtype=np.float32)  # (1, 2, 3)

                link_paths = self.get_link_scene_paths()


                path = f"{link_paths[frame_name]}/force"

                if self.force_visualization_initied:
                    self.force_handles[frame_name].points = points
                    self.force_handles[frame_name].visible = True
                else:
                    self.force_handles[frame_name] = self.server.scene.add_line_segments(
                        path, points, colors=np.array([255, 50, 50], dtype=np.uint8), line_width=3.0,
                    )


BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
sys.path.insert(0, str(BUILD_DIR))

import sqp_solver as sqp
import pinocchio as pin

RESOURCES = Path(__file__).resolve().parent.parent / "resources"

def main():
    dt = 0.02
    feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    scheduler = sqp.ContactScheduler()
    scheduler.define_contact("FL_RR", ["FL_foot", "RR_foot"])
    scheduler.define_contact("FR_RL", ["FR_foot", "RL_foot"])

    stance_duration = 0.1;  # 150ms per diagonal stance
    scheduler.addPhase(["FR_RL"], stance_duration, "trot")         # FR+RL stance
    scheduler.addPhase(["FL_RR"], stance_duration, "trot")         # FR+RL stance

    # Generate contact sequence for 2 full gait cycles
    N = 20;
    contact_sequence = scheduler.getSequence(sampling_rate=dt, sequence_name="trot", nodes_number=N)

    print(f"Horizon: N={N}  nodes={N}   T={N*dt}")
    print(f"Gait: trot ({stance_duration}s per diagonal stance)");

    urdf_path = str(RESOURCES / "urdf/go2/go2.urdf")
    print(f"URDF: {urdf_path}")
    rviz = rvizer(urdf_path)

    # ── Nominal standing configuration ──
    # Floating base: [x, y, z, qx, qy, qz, qw] + 12 joint angles
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
    q0 = pin.neutral(model)
    q0[2] = 0.30  # base height ~30cm

    # Hip, thigh, calf angles for a standing pose
    # FL
    q0[7]  =  0.0;    #FL_hip
    q0[8]  =  0.8;    # FL_thigh
    q0[9]  = -1.6;    # FL_calf
    # FR
    q0[10] =  0.0;    # FR_hip
    q0[11] =  0.8;    # FR_thigh
    q0[12] = -1.6;    # FR_calf
    # RL
    q0[13] =  0.0;    # RL_hip
    q0[14] =  0.8;    # RL_thigh
    q0[15] = -1.6;    # RL_calf
    # RR
    q0[16] =  0.0;    # RR_hip
    q0[17] =  0.8;    # RR_thigh
    q0[18] = -1.6;    # RR_calf

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


    # ── Build OCP ──
    ocp = sqp.OCP(N)
    base_velocity = []
    for k in range(N - 1):
        model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
        node = sqp.Node(model)

        for foot in feet:
            node.add_contact(foot)
        node.set_active_contacts(contact_sequence[k])

        node.add_dynamics(sqp.SemiEulerIntegration(dt))
        node.add_constraint(sqp.InvDynamics())

        # Contact and friction constraints for each foot
        for foot in feet:
            node.add_constraint(sqp.ContactConstraint(foot));
            node.add_constraint(sqp.FrictionConeConstraint(foot, 0.8))


        # Costs
        base_velocity.append(sqp.FrameVelocityCost("base", weight=50.))
        node.add_cost(base_velocity[k])

        node.add_cost(sqp.ConfigurationCost(q0, 10.))
        node.add_cost(sqp.TorqueCost(1e-4))

        node.add_cost(sqp.VelocityCost(1e-6))
        node.add_cost(sqp.AccelerationCost(1e-9))


        ocp.addNode(node)

    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
    node = sqp.Node(model)
    for foot in feet:
        node.add_contact(foot)
    node.set_active_contacts(contact_sequence[k])

    base_velocity.append(sqp.FrameVelocityCost("base", weight=50.))
    node.add_cost(base_velocity[-1])

    node.add_cost(sqp.VelocityCost(1e-6))
    node.add_cost(sqp.ConfigurationCost(q0, 10.))

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
    opts.tolerance     = 5e-3
    #opts.ls_merit_eta  = 1e-4
    opts.ls_type       = sqp.LSType.MERIT
    solver.set_options(opts)

    solver.solve()

    #ocp.save_trajectory("/workspace/code/resources/trajectories/go2_mpc.json", dt, urdf_path)

    opts = sqp.SQPoptions()
    opts.max_sqp_iters = 1
    opts.verbose = 0
    opts.ls_type = sqp.LSType.NONE
    solver.set_options(opts)

    t = 0.
    contact_forces = {}
    for foot in feet:
        contact_forces[foot] = np.zeros((3,1))
    while True:
        vx, vy, wz = get_velocity()

        # send to robot
        for bv in base_velocity:
            bv.set_ref(np.array([vx, vy, 0., 0., 0., wz]))

        # Update contact schedule
        contact_sequence = scheduler.getSequence(dt, "trot", N, t);
        for k in range(N):
            ocp.get_node(k).set_active_contacts(contact_sequence[k]);

        solver.solve(ocp.x_traj()[1][:])
        t+=dt

        q1 = ocp.x_traj()[1][0:model.nq]
        v1 = ocp.x_traj()[1][model.nq:]

        u0 = ocp.u_traj()[0]
        i = 0
        for foot in feet:
            contact_forces[foot] = u0[model.nv+i*3:model.nv+(i+1)*3]
            i += 1

        pos = np.array(q1[:3])
        quat_xyzw = q1[3:7]

        with rviz.server.atomic():
            rviz.base_frame.position = pos
            rviz.base_frame.wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            rviz.viser_urdf.update_cfg(q1[7:])
            rviz.update_forces(contact_forces, scale = 0.01)
        rviz.server.flush()

        time.sleep(0.01)


if __name__ == "__main__":
    main()