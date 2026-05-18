import numpy as np

class ReferenceInterpolator:
    def __init__(self, dt_mpc, dt_ctrl):
        self.dt_mpc = dt_mpc
        self.dt_ctrl = dt_ctrl

        self.counter = 0
        self.steps_per_mpc = int(round(dt_mpc / dt_ctrl))

        self.q0 = None
        self.v0 = None
        self.q1 = None
        self.v1 = None

    def update_mpc_plan(self, q0, v0, q1, v1):
        self.q0 = q0.copy()
        self.v0 = v0.copy()

        self.q1 = q1.copy()
        self.v1 = v1.copy()

        self.counter = 0

    def get_references(self):
        alpha = self.counter / self.steps_per_mpc
        alpha = min(max(alpha, 0.0), 1.0)

        q_ref = (1.0 - alpha) * self.q0 + alpha * self.q1
        v_ref = (1.0 - alpha) * self.v0 + alpha * self.v1

        self.counter += 1

        return q_ref, v_ref