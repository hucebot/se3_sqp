import mujoco
import glfw
import numpy as np
import sys
from pathlib import Path
from joint_impedance_ctrl import JointImpedanceController, convert, to_pinocchio_qpos, to_mujoco_qpos
from mujoco_viewer import MujocoViewer

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
# Load models (mujoco and pinocchio)
# -----------------------------
MODEL_PATH = str(RESOURCES / "mjcf/go2/scene.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

urdf_path = str(RESOURCES / "urdf/go2/go2.urdf")
pin_model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())

mujoco_joint_names = []
for a in range(model.nu):
    actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
    joint_id = model.actuator_trnid[a][0]
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    mujoco_joint_names.append(joint_name)

pinocchio_joint_names = list(pin_model.names)[2:]
print(f"mujoco_joint_names: {mujoco_joint_names}")
print(f"pinocchio_joint_names: {pinocchio_joint_names}")

# -----------------------------
# INITIAL CONFIGURATION
# -----------------------------
q0 = pin.neutral(pin_model)
q0[2] = 0.3
q0[7:] = np.array([0.0, 0.8, -1.6,  # hip, thigh, calf
                   0.0, 0.8, -1.6,  # hip, thigh, calf
                   0.0, 0.8, -1.6,  # hip, thigh, calf
                   0.0, 0.8, -1.6]) # hip, thigh, calf
data.qpos = to_mujoco_qpos(mujoco_joint_names, pinocchio_joint_names, q0)

# velocities
data.qvel[:] = 0.0

# propagate state
mujoco.mj_forward(model, data)

# -----------------------------
# Main loop
# -----------------------------
dt = 0.01
model.opt.timestep = dt/2.
feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

scheduler = sqp.ContactScheduler()
scheduler.define_contact("FL_RR", ["FL_foot", "RR_foot"])
scheduler.define_contact("FR_RL", ["FR_foot", "RL_foot"])

stance_duration = 0.15  # 150ms per diagonal stance
scheduler.addPhase(["FR_RL"], stance_duration, "trot")         # FR+RL stance
scheduler.addPhase(["FL_RR"], stance_duration, "trot")         # FR+RL stance

# Generate contact sequence for 2 full gait cycles
N = 20;
contact_sequence = scheduler.getSequence(sampling_rate=dt, sequence_name="trot", nodes_number=N)

print(f"Horizon: N={N}  nodes={N}   T={N*dt}")
print(f"Gait: trot ({stance_duration}s per diagonal stance)");


rviz = rviser.rviser(urdf_path)

# ── Nominal standing configuration ──
pin_data = pin.Data(pin_model)
pin.forwardKinematics(pin_model, pin_data, q0)
pin.updateFramePlacements(pin_model, pin_data)

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


joint_controller = JointImpedanceController(model, mujoco_joint_names)


# -----------------------------
# Mujoco Viewer
# -----------------------------
viewer = MujocoViewer(model)

t = 0.
joy = joystick.Joystick()
joy.start()
render_every = 1
counter = 0
try:
    while not viewer.should_close():
        # get joystick commands
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

        tau_ref = joint_controller.compute(data, q_des=convert(mujoco_joint_names, pinocchio_joint_names, q1[7:]),
        dq_des=convert(mujoco_joint_names, pinocchio_joint_names, v1[6:]),
        tau_ff=convert(mujoco_joint_names, pinocchio_joint_names, ocp.get_node(0).tau()[6:]))

        data.ctrl[:] = tau_ref

        mujoco.mj_step(model, data)

        #ocp.get_node(0).q()[7:] = pinocchio_to_mujoco(pinocchio_joint_names, mujoco_joint_names, data.qpos[7:])
        #ocp.get_node(0).v()[6:] = pinocchio_to_mujoco(pinocchio_joint_names, mujoco_joint_names, data.qvel[6:])
        #ocp.x_traj()[0][:] = np.concatenate([ocp.get_node(0).q(), ocp.get_node(0).v()])

        if counter % render_every == 0:
            viewer.render(data)

            rviz.update(q=q1[7:], q_base=q1[0:7])

        counter += 1

finally:
    viewer.close()
    glfw.terminate()
