import viser
from yourdfpy import URDF
from viser.extras import ViserUrdf
import numpy as np

class rviser:
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

        self.force_visualization_initied = False
        self.force_handles = {}

    def get_link_scene_paths(self) -> dict[str, str]:
        """Map link names to their viser scene paths by inspecting joint frames."""
        paths = {}
        for joint, frame_handle in zip(
            self.viser_urdf._joint_map_values, self.viser_urdf._joint_frames
        ):
            paths[joint.child] = frame_handle.name
        return paths

    def update_forces(self, contact_forces: dict, scale: float = 0.05) -> None:
            """Draw contact force arrows in the contact frames."""
            for frame_name, force_local in contact_forces.items():
                f_local = np.array(force_local, dtype=np.float32)
                f_norm = float(np.linalg.norm(f_local))

                # Line segment from origin to scaled force, in the contact frame
                tip = scale * f_local
                points = np.array([[[0, 0, 0], tip]], dtype=np.float32)  # (1, 2, 3)

                link_paths = self.get_link_scene_paths()


                path = f"{link_paths[frame_name]}/force"

                if self.force_visualization_initied:
                    self.force_handles[frame_name].points = points
                    self.force_handles[frame_name].visible = True
                else:
                    self.force_handles[frame_name] = self.server.scene.add_line_segments(
                        path, points, colors=np.array([255, 50, 50], dtype=np.uint8), line_width=3.0,
                    )
            self.force_visualization_initied = True

    def update(self, q=None, q_base=None, forces=None) -> None:
        with self.server.atomic():
            if q_base is not None:
                self.base_frame.position = q_base[0:3]
                self.base_frame.wxyz = np.array([q_base[6], q_base[3], q_base[4], q_base[5]])
            if q is not None:
                self.viser_urdf.update_cfg(q)
            if forces is not None:
                self.update_forces(forces, scale = 0.01)
        self.server.flush()