import mujoco
import numpy as np

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

    def set_kp(self, kp):
        if isinstance(kp, float):
            self.kp = np.ones(self.nu) * kp
        else:
            self.kp = kp

    def set_kd(self, kd):
        if isinstance(kd, float):
            self.kd = np.ones(self.nu) * kd
        else:
            self.kd = kd

    def compute(self, data, q_des, dq_des=None, tau_ff=None):
        if dq_des is None:
            dq_des = np.zeros(self.nu)

        if tau_ff is None:
            tau_ff = np.zeros(self.nu)

        q = data.qpos[self.q_idx]
        dq = data.qvel[self.v_idx]

        return self.kp * (q_des - q) + self.kd * (dq_des - dq) + tau_ff
