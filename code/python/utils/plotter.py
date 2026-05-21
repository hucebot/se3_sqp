import viser
import numpy as np
from collections.abc import Iterable

class plot:
    def __init__(self, title, size, legend_label, server, dt, max_len=1000):
        self.size = size
        self.max_len = max_len
        self.dt = dt

        # Normalize legend labels
        if isinstance(legend_label, str):
            labels = [f"{legend_label}{i}" for i in range(size)]

        elif isinstance(legend_label, Iterable):
            labels = list(legend_label)

            if len(labels) != size:
                raise ValueError(
                    f"Expected {size} legend labels, got {len(labels)}"
                )
        else:
            raise TypeError(
                "legend_label must be either a string or an iterable of strings"
            )

        # One shared time axis
        self.x = self.dt * np.arange(max_len, dtype=np.float64)

        # Y series
        self.ys = [np.zeros(max_len, dtype=float) for _ in range(size)]

        # uPlot data format
        self.data = (self.x, *self.ys)

        # Series definition
        self.series = (
            viser.uplot.Series(label="t"),
            *[
                viser.uplot.Series(
                    label=labels[i],
                    stroke=self.make_color(i, size),
                    width=2,
                )
                for i in range(size)
            ]
        )

        self.plot_handle = server.gui.add_uplot(
            data=self.data,
            series=self.series,
            title=title,
            scales={
                "x": viser.uplot.Scale(time=False, auto=True),
                "y": viser.uplot.Scale(auto=True),
            },
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



class plot_solver_stats:
    def __init__(self, server, dt, number_of_nodes):
        self.iteration_legend = ["SQP", "LS", "QP"]
        self.number_of_iterations = plot(title="Iterations", size=3, legend_label=self.iteration_legend, server=server, dt=dt)

        self.sqp_time_legend = ["tot", "last"]
        self.sqp_time = plot(title="SQP Timings [ms]", size=2, legend_label=self.sqp_time_legend, server=server, dt=dt)

        self.total_cost = plot(title="SQP Cost", size=1, legend_label="total", server=server, dt=dt)

        self.total_constraint_violation = plot(title="SQP Constraint Violation", size=1, legend_label="total", server=server, dt=dt)

        self.per_node_violation = plot(title="SQP Constraint Violation", size=number_of_nodes, legend_label="node", server=server, dt=dt)

        self.total_dynamics_defect = plot(title="SQP Dynamics Defect", size=1, legend_label="total", server=server, dt=dt)

        self.convergence_legend = ["sn", "di"]
        self.convergence = plot(title="SQP Convergence", size=2, legend_label=self.convergence_legend, server=server, dt=dt)

    def update(self, solver_stats):
        self.number_of_iterations.update(np.array([solver_stats.number_of_iterations, solver_stats.linesearch_iterations, solver_stats.qp_iterations]))
        self.sqp_time.update(np.array([solver_stats.total_time_ms, solver_stats.last_iteration_time_ms]))
        self.total_cost.update(np.array([solver_stats.total_cost]))
        self.total_constraint_violation.update(np.array([solver_stats.total_constraint_violation]))

        if len(solver_stats.per_node_violation) > 0:
            self.per_node_violation.update(solver_stats.per_node_violation)

            self.total_dynamics_defect.update(np.array([solver_stats.total_dynamics_defect]))
        self.convergence.update(np.array([solver_stats.step_norm, solver_stats.dual_infeasibility]))
