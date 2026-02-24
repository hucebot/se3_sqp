"""Python port of examples/example_trajopt.cpp

Swings a pendubot (double pendulum) from the downward rest position to
the upright balanced configuration using SQP trajectory optimization.

Usage:
    python3 example_trajopt.py

Requires the sqp_solver Python module to be built first:
    cmake -B build -DBUILD_PYTHON_BINDINGS=ON -DBUILD_EXAMPLES=OFF
    cmake --build build --target sqp_solver_py -j$(nproc)

Then either run from the build directory or set PYTHONPATH:
    PYTHONPATH=/workspace/code/build python3 python/example_trajopt.py
"""

import sys
import math
from pathlib import Path

import numpy as np

BUILD_DIR = Path(__file__).resolve().parent.parent / "build"
sys.path.insert(0, str(BUILD_DIR))

import sqp_solver as sqp

RESOURCES = Path(__file__).resolve().parent.parent / "resources"

def main():
    N  = 100
    dt = 0.01

    urdf_path = str(RESOURCES / "pendubot.urdf")
    print(f"URDF: {urdf_path}")

    ocp = sqp.OCP(N)

    q_ref = np.array([math.pi, 0.0])  # upright configuration

    # Running nodes
    for _ in range(N - 1):
        node = sqp.Node(urdf_path)
        node.add_dynamics(sqp.SemiEulerIntegration(dt))
        node.add_constraint(sqp.InvDynamics())
        node.add_cost(sqp.ConfigurationCost(q_ref, 0.0))
        node.add_cost(sqp.VelocityCost(1e-6))
        node.add_cost(sqp.AccelerationCost(1e-9))
        ocp.addNode(node)

    # Terminal node — high weight on reaching the target
    node = sqp.Node(urdf_path)
    node.add_cost(sqp.ConfigurationCost(q_ref, 1e0))
    node.add_cost(sqp.VelocityCost(1e0))
    ocp.addNode(node)

    ocp.finalize()

    # Initial guess: downward rest position, zero velocity and control
    for k in range(N):
        ocp.get_node(k).q()[:] = [0.0, 0.0]
        ocp.get_node(k).v()[:] = 0.0
        ocp.get_node(k).u()[:] = 0.0

    # Solve
    solver = sqp.SQPSolver(ocp)
    opts = sqp.SQPoptions()
    opts.max_sqp_iters = 1000
    opts.tolerance     = 1e-3
    opts.ls_merit_eta  = 1e-4
    opts.ls_type       = sqp.LSType.MERIT
    solver.set_options(opts)

    solver.solve()


if __name__ == "__main__":
    main()
