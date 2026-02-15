"""
Visualize a trajectory saved from the C++ SQP solver.

Usage:
    python3 visualize_trajectory.py [path/to/trajectory.json]

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

import json
import os
import sys
import time

import numpy as np
import viser
from pathlib import Path
from viser.extras import ViserUrdf



def main(traj_path: str = "/workspace/trajectory.json") -> None:
    with open(traj_path) as f:
        data = json.load(f)

    N         = data["N"]
    dt        = data["dt"]
    nq        = data["nq"]
    urdf_path = data["urdf_path"]
    traj      = data["trajectory"]          # list of {"q", "v", "u"} dicts

    server = viser.ViserServer()
    print(f"Viser listening at http://localhost:8080")



    viser_robot = ViserUrdf(server, urdf_or_path=Path(urdf_path))

    # GUI controls
    with server.gui.add_folder("Playback"):
        speed_slider = server.gui.add_slider(
            "Speed", min=0.1, max=10.0, step=0.1, initial_value=1.0
        )
        playing = server.gui.add_checkbox("Play", initial_value=True)

    print(f"Replaying {N} timesteps (dt={dt}s). Open browser at http://localhost:8080")

    k = 0
    while True:
        if playing.value:
            q = np.array(traj[k]["q"])
            viser_robot.update_cfg(q)
            k = (k + 1) % N
            time.sleep(dt / speed_slider.value)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/trajectory.json"
    main(path)
