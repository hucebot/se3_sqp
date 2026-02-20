"""Visualize a trajectory saved from the C++ SQP solver.

Usage:
    python3 visualize_trajectory.py [--trajectory path/to/trajectory.json]

The trajectory JSON is produced by OCP::save_trajectory() and has the form:
    {
        "N": <int>,
        "dt": <float>,
        "nq": <int>,
        "nv": <int>,
        "nu": <int>,
        "urdf_path": <str>,
        "trajectory": [{"q": [...], "v": [...], "u": [...]}, ...]
    }
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import viser
from viser.extras import ViserUrdf

RESOURCES = Path(__file__).resolve().parent.parent / "resources"
TRAJ_DIR  = RESOURCES / "trajectories"
URDF_DIR  = RESOURCES


def find_files() -> tuple[dict[str, Path], dict[str, Path]]:
    trajs = {p.name: p for p in sorted(TRAJ_DIR.rglob("*.json"))} if TRAJ_DIR.exists() else {}
    urdfs = {p.name: p for p in sorted(URDF_DIR.rglob("*.urdf"))}
    return trajs, urdfs


def load_trajectory(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def is_floating_base(data: dict) -> bool:
    """Floating base robots have nq = nv + 1 (quaternion orientation)."""
    return data["nq"] == data["nv"] + 1


FORCE_SCALE = 0.005  # m per N
FORCE_COLOR = np.array([255, 50, 50], dtype=np.uint8)


def get_link_scene_paths(viser_robot: ViserUrdf) -> dict[str, str]:
    """Map link names to their viser scene paths by inspecting joint frames."""
    paths = {}
    for joint, frame_handle in zip(
        viser_robot._joint_map_values, viser_robot._joint_frames
    ):
        paths[joint.child] = frame_handle.name
    return paths


def apply_q(viser_robot: ViserUrdf, base_frame, q: list, floating: bool) -> None:
    """Set the robot configuration from a full q vector."""
    if floating:
        # q = [x, y, z, qx, qy, qz, qw, joint0, joint1, ...]
        pos = np.array(q[:3])
        quat_xyzw = q[3:7]
        joints = np.array(q[7:])
        # viser uses wxyz quaternion convention
        base_frame.position = pos
        base_frame.wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        viser_robot.update_cfg(joints)
    else:
        viser_robot.update_cfg(np.array(q))


def update_forces(
    server: viser.ViserServer,
    link_paths: dict[str, str],
    force_handles: dict,
    contact_forces: dict,
    scale: float = FORCE_SCALE,
) -> None:
    """Draw contact force arrows in the contact frames."""
    for frame_name, force_local in contact_forces.items():
        f_local = np.array(force_local, dtype=np.float32)
        f_norm = float(np.linalg.norm(f_local))

        path = f"{link_paths[frame_name]}/force"

        if f_norm < 1e-3:
            if path in force_handles:
                force_handles[path].visible = False
            continue

        # Line segment from origin to scaled force, in the contact frame
        tip = scale * f_local
        points = np.array([[[0, 0, 0], tip]], dtype=np.float32)  # (1, 2, 3)

        if path in force_handles:
            force_handles[path].points = points
            force_handles[path].visible = True
        else:
            force_handles[path] = server.scene.add_line_segments(
                path, points, colors=FORCE_COLOR, line_width=3.0,
            )


def main(traj_path: Path) -> None:
    data = load_trajectory(traj_path)

    floating = is_floating_base(data)
    state = {
        "N":        data["N"],
        "dt":       data["dt"],
        "traj":     data["trajectory"],
        "k":        0,
        "floating": floating,
    }
    urdf_path = Path(data["urdf_path"])

    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=True)

    # Scene decoration
    # server.scene.world_axes.visible = True
    server.scene.add_grid(
        "/floor",
        width=6.0,
        height=6.0,
        plane="xy",
        cell_size=0.25,
        section_size=1.0,
    )

    # Create a parent frame for the robot base (used for floating base transforms)
    base_frame = [server.scene.add_frame("/robot_base", show_axes=False)]
    viser_robot = [ViserUrdf(server, urdf_or_path=urdf_path, root_node_name="/robot_base")]
    link_paths = [get_link_scene_paths(viser_robot[0])]
    force_handles: dict = {}

    # Playback controls
    with server.gui.add_folder("Playback"):
        playing      = server.gui.add_checkbox("Play", initial_value=True)
        speed_slider = server.gui.add_slider(
            "Speed", min=0.1, max=2.0, step=0.1, initial_value=1.0
        )
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=state["N"] - 1, step=1, initial_value=0
        )

    # Force visualization
    has_forces = "contact_forces" in state["traj"][0]
    with server.gui.add_folder("Forces"):
        show_forces = server.gui.add_checkbox("Show forces", initial_value=has_forces)
        force_scale_slider = server.gui.add_slider(
            "Scale", min=0.001, max=0.02, step=0.001, initial_value=FORCE_SCALE
        )

    # File selection
    traj_files, urdf_files = find_files()
    with server.gui.add_folder("Files"):
        traj_dropdown = server.gui.add_dropdown(
            "Trajectory",
            options=list(traj_files) or [traj_path.name],
            initial_value=traj_path.name,
        )
        urdf_dropdown = server.gui.add_dropdown(
            "URDF",
            options=list(urdf_files) or [urdf_path.name],
            initial_value=urdf_path.name,
        )

    # Callbacks
    @traj_dropdown.on_update
    def _(_) -> None:
        try:
            new_data          = load_trajectory(traj_files[traj_dropdown.value])
            state["N"]        = new_data["N"]
            state["dt"]       = new_data["dt"]
            state["traj"]     = new_data["trajectory"]
            state["floating"] = is_floating_base(new_data)
            state["k"]        = 0
            frame_slider.value = 0
        except Exception as e:
            print(f"[error] failed to load trajectory: {e}")

    @urdf_dropdown.on_update
    def _(_) -> None:
        try:
            viser_robot[0].remove()
            viser_robot[0] = ViserUrdf(
                server, urdf_or_path=urdf_files[urdf_dropdown.value],
                root_node_name="/robot_base",
            )
            link_paths[0] = get_link_scene_paths(viser_robot[0])
            force_handles.clear()
        except Exception as e:
            print(f"[error] failed to load URDF: {e}")

    @frame_slider.on_update
    def _(_) -> None:
        idx = min(frame_slider.value, state["N"] - 1)
        try:
            state["k"] = idx
            node = state["traj"][idx]
            apply_q(viser_robot[0], base_frame[0], node["q"], state["floating"])
            if show_forces.value and "contact_forces" in node:
                update_forces(
                    server, link_paths[0], force_handles,
                    node["contact_forces"], force_scale_slider.value,
                )
            elif not show_forces.value:
                for h in force_handles.values():
                    h.visible = False
        except ValueError as e:
            print(f"[error] update_cfg: {e}")
            playing.value = False

    print(f"Loaded {state['N']} timesteps (dt={state['dt']:.4f} s).  Open http://localhost:8080")

    while True:
        if playing.value:
            k = state["k"]
            frame_slider.value = k
            state["k"] = (k + 1) % state["N"]
            time.sleep(state["dt"] / speed_slider.value)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a trajectory from the SQP solver."
    )
    parser.add_argument(
        "--trajectory", "-t",
        type=Path,
        default=TRAJ_DIR / "trajectory.json",
        help=f"Path to the trajectory JSON file (default: {TRAJ_DIR / 'trajectory.json'}).",
    )
    args = parser.parse_args()
    main(args.trajectory)
