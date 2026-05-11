from scipy.spatial.transform import Rotation as R
import numpy as np

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