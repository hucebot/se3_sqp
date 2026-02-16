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


def main(traj_path: Path) -> None:
    data = load_trajectory(traj_path)

    state = {
        "N":    data["N"],
        "dt":   data["dt"],
        "traj": data["trajectory"],
        "k":    0,
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

    viser_robot = [ViserUrdf(server, urdf_or_path=urdf_path)]

    # Playback controls
    with server.gui.add_folder("Playback"):
        playing      = server.gui.add_checkbox("Play", initial_value=True)
        speed_slider = server.gui.add_slider(
            "Speed", min=0.1, max=1.0, step=0.1, initial_value=1.0
        )
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=state["N"] - 1, step=1, initial_value=0
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
            new_data      = load_trajectory(traj_files[traj_dropdown.value])
            state["N"]    = new_data["N"]
            state["dt"]   = new_data["dt"]
            state["traj"] = new_data["trajectory"]
            state["k"]    = 0
            frame_slider.value = 0
        except Exception as e:
            print(f"[error] failed to load trajectory: {e}")

    @urdf_dropdown.on_update
    def _(_) -> None:
        try:
            viser_robot[0].remove()
            viser_robot[0] = ViserUrdf(server, urdf_or_path=urdf_files[urdf_dropdown.value])
        except Exception as e:
            print(f"[error] failed to load URDF: {e}")

    @frame_slider.on_update
    def _(_) -> None:
        idx = min(frame_slider.value, state["N"] - 1)
        try:
            viser_robot[0].update_cfg(np.array(state["traj"][idx]["q"]))
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
