import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np
import viser
from yourdfpy import URDF
from viser.extras import ViserUrdf
import time
import threading

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

class interactive_marker():
    def __init__(self, server, translation_task, orientation_task, lock):

        self.translation_task = translation_task
        self.orientation_task = orientation_task
        self.lock = lock

        translation = self.translation_task.get_ref()
        quat = R.from_matrix(self.orientation_task.get_ref()).as_quat()  # x y z w

        self.int_marker = server.scene.add_transform_controls(
            "/"+translation_task.get_name(),
            position=(translation[0], translation[1], translation[2]),
            wxyz=(quat[3], quat[0], quat[1], quat[2]),
            scale=0.5
        )
        self.int_marker.on_update(self.on_target_move())


    def on_target_move(self):
        def _(_):
            p = self.int_marker.position
            o = self.int_marker.wxyz

            new_translation = np.array(p)
            new_orientation = R.from_quat([o[1], o[2], o[3], o[0]])

            with self.lock:
                self.translation_task.set_ref(new_translation)
                self.orientation_task.set_ref(new_orientation.as_matrix())
        return _

class plot:
    def __init__(self, title, size, legend_label, server, dt, max_len=1000):
        self.size = size
        self.max_len = max_len
        self.dt = dt

        # One shared time axis
        self.x = self.dt * np.arange(max_len, dtype=np.float64)

        # Y series: one array per joint
        self.ys = [np.zeros(max_len, dtype=float) for _ in range(size)]

        # uPlot data format: (x, y0, y1, ..., yN)
        self.data = (self.x, *self.ys)

        # Series definition: one series per array in data

        self.series = (
            viser.uplot.Series(label="t"),
            *[viser.uplot.Series(label=f"{legend_label}{i}", stroke=self.make_color(i, size), width=2) for i in range(size)])

        self.plot_handle = server.gui.add_uplot(
            data=self.data,
            series=self.series,
            title=title,
            scales={"x": viser.uplot.Scale(time=False, auto=True), "y": viser.uplot.Scale(auto=True)},
            legend=viser.uplot.Legend(show=True),
            aspect=2.0,
        )

    def make_color(self, i, n):
        # HSV equally spaced around the hue circle
        h = i / n
        s = 0.65
        v = 0.85
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"

    def update(self, new_data):
        # Append new data
        self.x = np.roll(self.x, -1)
        self.x[-1] = self.x[-2] + self.dt

        for i in range(self.size):
            self.ys[i] = np.roll(self.ys[i], -1)
            self.ys[i][-1] = new_data[i]

        self.plot_handle.data = (self.x.tolist(), *[y.tolist() for y in self.ys])



BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
sys.path.insert(0, str(BUILD_DIR))

import sqp_solver as sqp
import pinocchio as pin

RESOURCES = Path(__file__).resolve().parent.parent / "resources"

def main():
    T = 0.1
    N  = 10
    dt = T/N
    print(f"dt: {dt}")

    urdf_path = str(RESOURCES / "urdf/floating_frame.urdf")
    print(f"URDF: {urdf_path}")
    rviz = rvizer(urdf_path)


    ocp = sqp.OCP(N)

    # Initial configuration
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
    q0 = pin.neutral(model)
    print(f"q0: {q0}")

    data = pin.Data(model)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)

    # Running nodes
    v_lims = np.array([10., 10., 10., 10., 10., 10.])
    for k in range(N - 1):
        model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
        node = sqp.Node(model)

        node.add_dynamics(sqp.SemiEulerIntegration(dt))

        if k > 0:
            node.add_cost(sqp.FrameVelocityCost("base_link", np.array([0., 0., 0., 0., 0., 0.]), 1e-3))
            node.add_constraint(sqp.FrameVelocityConstraint("base_link", v_lims))

        node.add_cost(sqp.FrameAccelerationCost("base_link", np.array([0., 0., 0., 0., 0., 0.]), 1e-3))

        ocp.addNode(node)


    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path)
    node = sqp.Node(model)

    translation_task = sqp.FrameTranslationCost("base_link", q0[0:3], 1e3)
    orientation_task = sqp.FrameOrientationCost("base_link", R.from_quat(q0[3:]).as_matrix(), 1e3) # xyzw

    node.add_cost(translation_task)
    node.add_cost(orientation_task)
    node.add_cost(sqp.FrameVelocityCost("base_link", np.array([0., 0., 0., 0., 0., 0.]), 1e6))
    node.add_constraint(sqp.FrameVelocityConstraint("base_link", v_lims))
    ocp.addNode(node)

    ocp.finalize()

    # Initial guess: downward rest position, zero velocity and control
    for k in range(N-1):
        ocp.get_node(k).q()[:] = q0
        ocp.get_node(k).v()[:] = 0.0
        ocp.get_node(k).u()[:] = 0.0
    ocp.get_node(N-1).q()[:] = q0

        # Solve
    solver = sqp.SQPSolver(ocp)
    opts = sqp.SQPoptions()
    opts.max_sqp_iters = 100
    opts.hpipm_iter_max = 20
    opts.eps_inequality = 1e-4
    opts.ls_type       = sqp.LSType.MERIT
    opts.ls_merit_mu = 10.
    opts.ls_merit_eta = 1e-4
    solver.set_options(opts)

    solver.solve()

    pos = ocp.x_traj()[0][0:3]
    quat_xyzw = ocp.x_traj()[0][3:7]
    v = ocp.x_traj()[1][model.nq:]


    opts = sqp.SQPoptions()
    opts.max_sqp_iters = 1
    opts.verbose = 0
    opts.ls_type       = sqp.LSType.NONE
    solver.set_options(opts)

    # Markers
    lock = threading.Lock()
    interactive_marker(rviz.server, translation_task, orientation_task, lock)

    # Plot
    v_plot = plot(title="Velocities", size=model.nv, legend_label="v", server=rviz.server, dt=dt)

    while True:
        #ocp.x_traj()[0][:] = ocp.x_traj()[1][:]
        solver.solve(ocp.x_traj()[0][:])

        q1 = ocp.x_traj()[1][0:model.nq]
        v1 = ocp.x_traj()[1][model.nq:]

        #q1 = ocp.x_traj()[k][0:model.nq]
        pos = np.array(q1[:3])
        quat_xyzw = q1[3:7]

        rviz.base_frame.position = pos
        rviz.base_frame.wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        v_plot.update(v1)

        for k in range(N-2):
            ocp.get_node(k).q()[:] = ocp.get_node(k+1).q()[:]
            ocp.get_node(k).v()[:] = ocp.get_node(k+1).v()[:]
            ocp.get_node(k).u()[:] = ocp.get_node(k+1).u()[:]

        time.sleep(dt)


if __name__ == "__main__":
    main()
