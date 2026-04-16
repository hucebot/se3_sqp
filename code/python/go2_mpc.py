import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np
import viser
from yourdfpy import URDF
from viser.extras import ViserUrdf
import time
import threading
import pinocchio
import struct
import glob
import fcntl
import array

class JoystickJS0:
    """
    Minimal Linux joystick reader for /dev/input/js*.

    Provides:
        vx, vy, wz in [-1, 1]

    Axis mapping is detected automatically using JSIOCGAXMAP.
    """

    JS_EVENT_FORMAT = "IhBB"
    JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)

    JS_EVENT_BUTTON = 0x01
    JS_EVENT_AXIS   = 0x02

    JSIOCGAXES  = 0x80016a11
    JSIOCGAXMAP = 0x80406a32

    # Linux ABS axis codes
    ABS_X  = 0x00
    ABS_Y  = 0x01
    ABS_RX = 0x03
    ABS_RY = 0x04

    def __init__(self, device=None, deadzone=0.05):

        if device is None:
            devices = sorted(glob.glob("/dev/input/js*"))
            if not devices:
                raise RuntimeError("No joystick device found in /dev/input/js*")
            device = devices[0]

        self.device = device
        self.deadzone = deadzone

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

        self._running = False
        self._thread = None

        # axis indices (detected automatically)
        self.ax_left_x = None
        self.ax_left_y = None
        self.ax_right_x = None

        self._detect_axis_mapping()

    def _detect_axis_mapping(self):
        """Query joystick axis map from kernel."""
        try:
            with open(self.device, "rb") as js:

                # number of axes
                buf = array.array('B', [0])
                fcntl.ioctl(js, self.JSIOCGAXES, buf)
                num_axes = buf[0]

                # axis map
                buf = array.array('B', [0] * 0x40)
                fcntl.ioctl(js, self.JSIOCGAXMAP, buf)

                axis_map = buf[:num_axes]

                axis_index = {code: i for i, code in enumerate(axis_map)}

                self.ax_left_x  = axis_index.get(self.ABS_X)
                self.ax_left_y  = axis_index.get(self.ABS_Y)
                self.ax_right_x = axis_index.get(self.ABS_RX)

                print("[Joystick] axis mapping:")
                print("  left_x :", self.ax_left_x)
                print("  left_y :", self.ax_left_y)
                print("  right_x:", self.ax_right_x)

        except Exception as e:
            print(f"[Joystick] axis detection failed: {e}")

    def start(self):
        """Start background reader thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop reader thread."""
        self._running = False
        if self._thread:
            self._thread.join()

    def get(self, alpha_lin=1., alpha_ang=1.):
        """Return current command (vx, vy, wz)."""
        return alpha_lin*self.vx, alpha_lin*self.vy, alpha_ang*self.wz

    def _apply_deadzone(self, v):
        return 0.0 if abs(v) < self.deadzone else v

    def _loop(self):
        try:
            with open(self.device, "rb") as js:

                while self._running:

                    ev_buf = js.read(self.JS_EVENT_SIZE)

                    if not ev_buf:
                        time.sleep(0.001)
                        continue

                    _, value, etype, number = struct.unpack(
                        self.JS_EVENT_FORMAT, ev_buf
                    )

                    v = value / 32767.0
                    v = self._apply_deadzone(v)

                    if etype & self.JS_EVENT_AXIS:

                        if number == self.ax_left_x:
                            self.vy = v

                        elif number == self.ax_left_y:
                            self.vx = -v

                        elif number == self.ax_right_x:
                            self.wz = v

        except Exception as e:
            print(f"[Joystick] Error: {e}")
            self._running = False

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
            self.force_visualization_initied = True


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
    N = 10;
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
            node.add_cost(sqp.ForceCost(frame_name=foot, weight=1e-9))
            node.add_cost(sqp.StepCost(frame_name=foot, step_height_ref=0.05, ground_ref= 0.0, weight=2.))


        # Costs
        base_velocity.append(sqp.FrameVelocityCost("base", weight=2.))
        node.add_cost(base_velocity[k])

        node.add_cost(sqp.ConfigurationCost(q0, 1.))
        node.add_cost(sqp.TorqueCost(1e-4))

        node.add_cost(sqp.VelocityCost(1e-6))
        node.add_cost(sqp.AccelerationCost(1e-9))

        ocp.addNode(node)

    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
    node = sqp.Node(model)
    for foot in feet:
        node.add_contact(foot)
        node.add_cost(sqp.StepCost(frame_name=foot, step_height_ref=0.05, ground_ref= 0.0, weight=20.))

    node.set_active_contacts(contact_sequence[N-1])

    base_velocity.append(sqp.FrameVelocityCost("base", weight=5.))
    node.add_cost(base_velocity[-1])

    node.add_cost(sqp.VelocityCost(1e-6))
    node.add_cost(sqp.ConfigurationCost(q0, 1.))

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
    opts.max_sqp_iters = 10
    opts.tolerance     = 5e-3
    #opts.ls_merit_eta  = 1e-4
    opts.ls_type       = sqp.LSType.MERIT
    solver.set_options(opts)

    solver.solve()

    solver_mpc = sqp.SQPSolver(ocp=ocp, mode=sqp.hpipm_mode.BALANCE)

    opts = sqp.SQPoptions()
    opts.max_sqp_iters = 1
    opts.verbose = 0
    opts.hpipm_warm_start = True
    opts.hpipm_tol_eq = 1e-3
    opts.hpipm_tol_ineq = 1e-3
    opts.hpipm_tol_stat = 1e-3
    opts.hpipm_tol_comp = 1e-3
    opts.ls_type = sqp.LSType.MERIT
    solver_mpc.set_options(opts)

    t = 0.
    contact_forces = {}
    joy = JoystickJS0()
    joy.start()
    for foot in feet:
        contact_forces[foot] = np.zeros((3,1))
    while True:
        vx, vy, wz = joy.get(alpha_lin=0.5, alpha_ang=1.)

        # send to robot
        for bv in base_velocity:
            bv.set_ref(np.array([vx, vy, 0., 0., 0., wz]))



        # Update contact schedule
        contact_sequence = scheduler.getSequence(dt, "trot", N, t);
        for k in range(N):
            ocp.get_node(k).set_active_contacts(contact_sequence[k]);


        solver_mpc.solve(ocp.x_traj()[1][:])
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