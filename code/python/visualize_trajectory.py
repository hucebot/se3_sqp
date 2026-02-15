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


def main(traj_path: Path) -> None:
    with open(traj_path) as f:
        data = json.load(f)

    N         = data["N"]
    dt        = data["dt"]
    nq        = data["nq"]
    urdf_path = data["urdf_path"]
    traj      = data["trajectory"]   # list of {"q", "v", "u"} dicts

    server = viser.ViserServer()

    # Scene decoration
    server.scene.world_axes.visible = True
    server.scene.add_grid(
        "/floor",
        width=6.0,
        height=6.0,
        plane="xy",
        cell_size=0.25,
        section_size=1.0,
    )

    viser_robot = ViserUrdf(server, urdf_or_path=Path(urdf_path))

    # GUI controls
    with server.gui.add_folder("Playback"):
        playing      = server.gui.add_checkbox("Play", initial_value=True)
        speed_slider = server.gui.add_slider(
            "Speed", min=0.1, max=1.0, step=0.1, initial_value=1.0
        )
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=N - 1, step=1, initial_value=0
        )

    @frame_slider.on_update
    def _(_) -> None:
        q = np.array(traj[frame_slider.value]["q"])
        viser_robot.update_cfg(q)

    print(f"Replaying {N} timesteps (dt={dt:.4f} s).  Open http://localhost:8080")

    k = 0
    while True:
        if playing.value:
            frame_slider.value = k
            k = (k + 1) % N
            time.sleep(dt / speed_slider.value)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a trajectory from the SQP solver."
    )
    parser.add_argument(
        "--trajectory", "-t",
        type=Path,
        default=Path("/workspace/trajectory.json"),
        help="Path to the trajectory JSON file (default: /workspace/trajectory.json).",
    )
    args = parser.parse_args()
    main(args.trajectory)
