import sys
from pathlib import Path
import numpy as np
import time
from utils import joy as joystick
from utils import rviser


BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
sys.path.insert(0, str(BUILD_DIR))

import sqp_solver as sqp
import pinocchio as pin

RESOURCES = Path(__file__).resolve().parent.parent / "resources"

def main():
    dt = 0.01
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

    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
    node = sqp.Node(model)
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
    opts.print_per_node_violation = True
    opts.hpipm_warm_start = True
    opts.hpipm_tol_eq = 1e-3
    opts.hpipm_tol_ineq = 1e-3
    opts.hpipm_tol_stat = 1e-3
    opts.hpipm_tol_comp = 1e-3
    opts.ls_type = sqp.LSType.MERIT
    opts.eps_inequality = 1e-3
    solver_mpc.set_options(opts)

    t = 0.
    contact_forces = {}
    joy = joystick.Joystick()
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

        solver_mpc.solve(ocp.x_traj()[0][:])
        t+=dt

        q1 = ocp.x_traj()[1][0:model.nq]
        v1 = ocp.x_traj()[1][model.nq:]



        u0 = ocp.u_traj()[0]
        i = 0
        for foot in feet:
            contact_forces[foot] = u0[model.nv+i*3:model.nv+(i+1)*3]
            i += 1

        rviz.update(q=q1[7:], q_base=q1[0:7], forces=contact_forces)

        for k in range(N-2):
            ocp.get_node(k).q()[:] = ocp.get_node(k+1).q()[:]
            ocp.get_node(k).v()[:] = ocp.get_node(k+1).v()[:]
            ocp.get_node(k).u()[:] = ocp.get_node(k+1).u()[:]


        time.sleep(0.001)


if __name__ == "__main__":
    main()
