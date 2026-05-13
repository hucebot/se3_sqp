import mujoco
import glfw
import numpy as np
import sys
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent.parent / "../build"
sys.path.insert(0, str(BUILD_DIR))
RESOURCES = Path(__file__).resolve().parent.parent / "../resources"

import sqp_solver as sqp
import pinocchio as pin

UTILS_PATH = Path(__file__).resolve().parent.parent / "utils"
sys.path.insert(0, str(UTILS_PATH))
import joy as joystick
import rviser, plotter

# -----------------------------
# Load model (URDF via MjSpec)
# -----------------------------
MODEL_PATH = str(RESOURCES / "mjcf/go2/scene.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# -----------------------------
# INITIAL CONFIGURATION
# -----------------------------

# base pose
data.qpos[0:3] = np.array([0.0, 0.0, 0.3])
data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

print("nq:", model.nq)
print("nv:", model.nv)
print("nu:", model.nu)

# Hip, thigh, calf angles for a standing pose
# FL
data.qpos[7]  =  0.0;    # FL_hip
data.qpos[8]  =  0.8;    # FL_thigh
data.qpos[9]  = -1.6;    # FL_calf
# FR
data.qpos[10] =  0.0;    # FR_hip
data.qpos[11] =  0.8;    # FR_thigh
data.qpos[12] = -1.6;    # FR_calf
# RL
data.qpos[13] =  0.0;    # RL_hip
data.qpos[14] =  0.8;    # RL_thigh
data.qpos[15] = -1.6;    # RL_calf
# RR
data.qpos[16] =  0.0;    # RR_hip
data.qpos[17] =  0.8;    # RR_thigh
data.qpos[18] = -1.6;    # RR_calf

# velocities
data.qvel[:] = 0.0

# propagate state
mujoco.mj_forward(model, data)

# -----------------------------
# GLFW init
# -----------------------------
if not glfw.init():
    raise RuntimeError("GLFW init failed")

window = glfw.create_window(1200, 900, "MuJoCo", None, None)
glfw.make_context_current(window)

context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=1000)

# -----------------------------
# IMPORTANT: initialize free camera
# -----------------------------
mujoco.mjv_defaultFreeCamera(model, cam)

# -----------------------------
# Mouse state
# -----------------------------
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# -----------------------------
# Mouse callbacks
# -----------------------------
def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right

    button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS


def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    if not (button_left or button_middle or button_right):
        return

    width, height = glfw.get_window_size(window)

    if button_left:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
    elif button_right:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H
    elif button_middle:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM
    else:
        return

    mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene ,cam)

def scroll(window, xoffset, yoffset):
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)

# -----------------------------
# Register callbacks
# -----------------------------
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

class JointImpedanceController:
    def __init__(self, model, joint_names):

        self.model = model
        self.joint_names = joint_names

        self.nu = len(joint_names)

        # ---------------------------------------------------
        # Joint indexing
        # ---------------------------------------------------

        self.q_idx = []
        self.v_idx = []
        self.u_idx = []

        for name in joint_names:

            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

            self.q_idx.append(model.jnt_qposadr[joint_id])
            self.v_idx.append(model.jnt_dofadr[joint_id])

            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name
            )

            self.u_idx.append(actuator_id)

        self.q_idx = np.array(self.q_idx)
        self.v_idx = np.array(self.v_idx)
        self.u_idx = np.array(self.u_idx)

        # ---------------------------------------------------
        # Gains
        # ---------------------------------------------------

        self.kp = np.ones(self.nu) * 50.0
        self.kd = np.ones(self.nu) * 3.0

    def compute(self, data, q_des, dq_des=None, tau_ff=None):
        if dq_des is None:
            dq_des = np.zeros(self.nu)

        if tau_ff is None:
            tau_ff = np.zeros(self.nu)

        q = data.qpos[self.q_idx]
        dq = data.qvel[self.v_idx]

        return self.kp * (q_des - q) + self.kd * (dq_des - dq) + tau_ff

# -----------------------------
# Main loop
# -----------------------------
dt = 0.01
model.opt.timestep = dt/2.
feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

scheduler = sqp.ContactScheduler()
scheduler.define_contact("FL_RR", ["FL_foot", "RR_foot"])
scheduler.define_contact("FR_RL", ["FR_foot", "RL_foot"])

stance_duration = 0.15;  # 150ms per diagonal stance
scheduler.addPhase(["FR_RL"], stance_duration, "trot")         # FR+RL stance
scheduler.addPhase(["FL_RR"], stance_duration, "trot")         # FR+RL stance

# Generate contact sequence for 2 full gait cycles
N = 20;
contact_sequence = scheduler.getSequence(sampling_rate=dt, sequence_name="trot", nodes_number=N)

print(f"Horizon: N={N}  nodes={N}   T={N*dt}")
print(f"Gait: trot ({stance_duration}s per diagonal stance)");

urdf_path = str(RESOURCES / "urdf/go2/go2.urdf")
print(f"URDF: {urdf_path}")
rviz = rviser.rviser(urdf_path)

# ── Nominal standing configuration ──
# Floating base: [x, y, z, qx, qy, qz, qw] + 12 joint angles
pin_model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
q0 = pin.neutral(pin_model)
q0[0:3] = data.qpos[0:3].copy()
q0[3:6] = data.qpos[4:7].copy()
q0[6] = data.qpos[3]
q0[7:] = data.qpos[7:].copy()
pin_data = pin.Data(pin_model)
pin.forwardKinematics(pin_model, pin_data, q0)
pin.updateFramePlacements(pin_model, pin_data)

print(pin_model)

# ── Build OCP ──
ocp = sqp.OCP(N)
base_velocity = []
for k in range(N - 1):
    pin_model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
    node = sqp.Node(pin_model)

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

    node.add_constraint(sqp.JointLimitsConstraint())

    # Costs
    base_velocity.append(sqp.FrameVelocityCost("base", weight=2.))
    base_velocity[k].set_re_reference_frame(sqp.ReferenceFrame.LOCAL)
    node.add_cost(base_velocity[k])

    node.add_cost(sqp.ConfigurationCost(q0, 1e-3))
    node.add_cost(sqp.TorqueCost(1e-4))

    node.add_cost(sqp.VelocityCost(1e-6))
    node.add_cost(sqp.AccelerationCost(1e-9))

    ocp.addNode(node)

pin_model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
node = sqp.Node(pin_model)
for foot in feet:
    node.add_contact(foot)
node.set_active_contacts(contact_sequence[N-1])

for foot in feet:
    node.add_constraint(sqp.ContactConstraint(foot));
    node.add_cost(sqp.StepCost(frame_name=foot, step_height_ref=0.05, ground_ref= 0.0, weight=2.))

base_velocity.append(sqp.FrameVelocityCost("base", weight=5.))
base_velocity[-1].set_re_reference_frame(sqp.ReferenceFrame.LOCAL)
node.add_cost(base_velocity[-1])

node.add_constraint(sqp.JointLimitsConstraint())

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
opts.max_sqp_iters = 20
opts.tolerance     = 1e-3
#opts.ls_merit_eta  = 1e-4
opts.ls_type       = sqp.LSType.MERIT
solver.set_options(opts)

solver.solve()

solver_mpc = sqp.SQPSolver(ocp=ocp, mode=sqp.hpipm_mode.SPEED_ABS)

opts = sqp.SQPoptions()
opts.max_sqp_iters = 1
opts.verbose = 0
opts.print_per_node_violation = False
opts.hpipm_warm_start = True
opts.hpipm_tol_eq = 1e-3
opts.hpipm_tol_ineq = 1e-3
opts.hpipm_tol_stat = 1e-3
opts.hpipm_tol_comp = 1e-3
opts.ls_type = sqp.LSType.MERIT
opts.eps_inequality = 1e-3
solver_mpc.set_options(opts)


mujoco_joint_names = []
for a in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    joint_id = model.actuator_trnid[a][0]
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    mujoco_joint_names.append(joint_name)

pinocchio_joint_names = list(pin_model.names)[2:]
print(f"mujoco_joint_names: {mujoco_joint_names}")
print(f"pinocchio_joint_names: {pinocchio_joint_names}")

joint_controller = JointImpedanceController(model, mujoco_joint_names)

def pinocchio_to_mujoco(mujoco_joints, pinocchio_joints, u_pinocchio):
    # Map joint name -> index in Pinocchio vector
    pin_map = {
        name: i for i, name in enumerate(pinocchio_joints)
    }

    # Build MuJoCo-ordered vector
    u_mujoco = np.array([
        u_pinocchio[pin_map[name]]
        for name in mujoco_joints
    ])

    return u_mujoco

t = 0.
joy = joystick.Joystick()
joy.start()
render_every = 1
counter = 0
while not glfw.window_should_close(window):
    vx, vy, wz = joy.get(alpha_lin=0.5, alpha_ang=1.)

    # send to robot
    for bv in base_velocity:
        bv.set_ref(np.array([vx, vy, 0., 0., 0., wz]))

    # Update contact schedule
    contact_sequence = scheduler.getSequence(dt, "trot", N, t);
    for k in range(N):
        ocp.get_node(k).set_active_contacts(contact_sequence[k]);

    solver_mpc.solve(ocp.x_traj()[0][:])
    t+=dt

    q1 = ocp.x_traj()[1][0:pin_model.nq]
    v1 = ocp.x_traj()[1][model.nq:]

    for k in range(N-2):
        ocp.get_node(k).q()[:] = ocp.get_node(k+1).q()[:]
        ocp.get_node(k).v()[:] = ocp.get_node(k+1).v()[:]
        ocp.get_node(k).u()[:] = ocp.get_node(k+1).u()[:]

    tau_ref = joint_controller.compute(data, q_des=pinocchio_to_mujoco(mujoco_joint_names, pinocchio_joint_names, q1[7:]),
        dq_des=pinocchio_to_mujoco(mujoco_joint_names, pinocchio_joint_names, v1[6:]),
        tau_ff=pinocchio_to_mujoco(mujoco_joint_names, pinocchio_joint_names, ocp.get_node(0).tau()[6:]))

    data.ctrl[:] = tau_ref

    mujoco.mj_step(model, data)

    #ocp.get_node(0).q()[7:] = pinocchio_to_mujoco(pinocchio_joint_names, mujoco_joint_names, data.qpos[7:])
    #ocp.get_node(0).v()[6:] = pinocchio_to_mujoco(pinocchio_joint_names, mujoco_joint_names, data.qvel[6:])
    #ocp.x_traj()[0][:] = np.concatenate([ocp.get_node(0).q(), ocp.get_node(0).v()])

    if counter % render_every == 0:
        viewport = mujoco.MjrRect(0, 0, 1200, 900)

        mujoco.mjv_updateScene(
            model, data, opt, None, cam,
            mujoco.mjtCatBit.mjCAT_ALL, scene
        )

        mujoco.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

        rviz.update(q=q1[7:], q_base=q1[0:7])

    counter += 1

glfw.terminate()