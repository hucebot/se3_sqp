import numpy as np
from scipy.spatial.transform import Rotation

def convert(to_joints, from_joints, u_from):
    map = {name: i for i, name in enumerate(from_joints)}
    return np.array([u_from[map[name]] for name in to_joints])

def to_pinocchio_qpos(pinocchio_joints, mujoco_joints, qpos_mj):
    qpos_pin = qpos_mj.copy()
    qpos_pin[3:6] = qpos_mj[4:7]
    qpos_pin[6] = qpos_mj[3]
    qpos_pin[7:] = convert(pinocchio_joints, mujoco_joints, qpos_mj[7:])
    return qpos_pin

def to_mujoco_qpos(mujoco_joints, pinocchio_joints, qpos_pin):
    qpos_mj = qpos_pin.copy()
    qpos_mj[3] = qpos_pin[6]
    qpos_mj[4:7] = qpos_pin[3:6]
    qpos_mj[7:] = convert(mujoco_joints, pinocchio_joints, qpos_pin[7:])
    return qpos_mj

def to_pinocchio_qvel(pinocchio_joints, mujoco_joints, qpos_mj, qvel_mj):
    v_pin = qvel_mj.copy()
    v_pin[6:] = convert(pinocchio_joints, mujoco_joints, qvel_mj[6:])

    qw, qx, qy, qz = qpos_mj[3:7]

    R_wb = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    R_bw = R_wb.T

    v_world = qvel_mj[0:3]
    w_local = qvel_mj[3:6]

    v_local = R_bw @ v_world

    v_pin[0:3] = v_local
    v_pin[3:6] = w_local

    return v_pin

def pinocchio_to_mujoco(mujoco_joints, pinocchio_joints, qpos_pin, qvel_pin):
    v_mj = qvel_pin.copy()
    v_mj[6:] = convert(mujoco_joints, pinocchio_joints, qvel_pin[6:])

    qx, qy, qz, qw = qpos_pin[3:7]

    R_wb = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

    v_local = qvel_pin[0:3]
    w_local = qvel_pin[3:6]

    v_world = R_wb @ v_local

    v_mj[0:3] = v_world
    v_mj[3:6] = w_local

    return v_mj