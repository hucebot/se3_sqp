import mujoco
import numpy as np

def convert(to_joints, from_joints, u_from):
    map = {name: i for i, name in enumerate(from_joints)}
    return np.array([u_from[map[name]] for name in to_joints])

def to_pinocchio_qpos(pinocchio_joints, mujoco_joints, mujoco_qpos):
    pinocchio_qpos = mujoco_qpos.copy()
    pinocchio_qpos[3:6] = mujoco_qpos[4:7]
    pinocchio_qpos[6] = mujoco_qpos[3]
    pinocchio_qpos[7:] = convert(pinocchio_joints, mujoco_joints, mujoco_qpos[7:])
    return pinocchio_qpos

def to_mujoco_qpos(mujoco_joints, pinocchio_joints, pinocchio_qpos):
    mujoco_qpos = pinocchio_qpos.copy()
    mujoco_qpos[3] = pinocchio_qpos[6]
    mujoco_qpos[4:7] = pinocchio_qpos[3:6]
    mujoco_qpos[7:] = convert(mujoco_joints, pinocchio_joints, pinocchio_qpos[7:])
    return mujoco_qpos

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

            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

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
