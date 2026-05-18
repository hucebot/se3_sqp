import struct
import glob
import fcntl
import array
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation

class Joystick:
    """
    Minimal Linux joystick reader for /dev/input/js*.

    Provides:
        vx, vy, wz in [-1, 1]

    Axis mapping is detected automatically using JSIOCGAXMAP.
    """

    JS_EVENT_FORMAT = "IhBB"
    JS_EVENT_SIZE = struct.calcsize(JS_EVENT_FORMAT)

    JS_EVENT_BUTTON = 0x01
    JS_EVENT_AXIS   = 0x02

    JSIOCGAXES  = 0x80016a11
    JSIOCGAXMAP = 0x80406a32

    # Linux ABS axis codes
    ABS_X  = 0x00
    ABS_Y  = 0x01
    ABS_RX = 0x03
    ABS_RY = 0x04
    ABS_Z  = 0x02
    ABS_RZ = 0x05

    RIGHT_X_CANDIDATES = [ABS_RX, ABS_Z, ABS_RZ]

    def __init__(self, device=None, deadzone=0.05):

        if device is None:
            devices = sorted(glob.glob("/dev/input/js*"))
            if not devices:
                raise RuntimeError("No joystick device found in /dev/input/js*")
            device = devices[0]

        self.device = device
        self.deadzone = deadzone

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

        self._running = False
        self._thread = None

        # axis indices (detected automatically)
        self.ax_left_x = None
        self.ax_left_y = None
        self.ax_right_x = None

        self._detect_axis_mapping()

    def _detect_axis_mapping(self):
        """Query joystick axis map from kernel."""
        try:
            with open(self.device, "rb") as js:

                # number of axes
                buf = array.array('B', [0])
                fcntl.ioctl(js, self.JSIOCGAXES, buf)
                num_axes = buf[0]

                # axis map
                buf = array.array('B', [0] * 0x40)
                fcntl.ioctl(js, self.JSIOCGAXMAP, buf)

                axis_map = buf[:num_axes]

                axis_index = {code: i for i, code in enumerate(axis_map)}

                self.ax_left_x = axis_index.get(self.ABS_X)
                self.ax_left_y = axis_index.get(self.ABS_Y)

                # detect right stick X among candidates
                self.ax_right_x = None
                for code in self.RIGHT_X_CANDIDATES:
                    if code in axis_index:
                        self.ax_right_x = axis_index[code]
                        break

                print("[Joystick] axis mapping:")
                print("  axis_map :", axis_map)
                print("  left_x   :", self.ax_left_x)
                print("  left_y   :", self.ax_left_y)
                print("  right_x  :", self.ax_right_x)

        except Exception as e:
            print(f"[Joystick] axis detection failed: {e}")

    def start(self):
        """Start background reader thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop reader thread."""
        self._running = False
        if self._thread:
            self._thread.join()

    def get(self, base_quat=np.array([0., 0., 0., 1.]), alpha_lin=1., alpha_ang=1.):

        self.vx, self.vy, self.vz, self.wz = self.planar_local_velocity_reference(self.vx, self.vy, self.wz, base_quat)

        return alpha_lin*self.vx, alpha_lin*self.vy, alpha_lin*self.vz, alpha_ang*self.wz

    def _apply_deadzone(self, v):
        return 0.0 if abs(v) < self.deadzone else v

    def _loop(self):
        try:
            with open(self.device, "rb") as js:

                while self._running:

                    ev_buf = js.read(self.JS_EVENT_SIZE)

                    if not ev_buf:
                        time.sleep(0.001)
                        continue

                    _, value, etype, number = struct.unpack(
                        self.JS_EVENT_FORMAT, ev_buf
                    )

                    v = value / 32767.0
                    v = self._apply_deadzone(v)

                    if etype & self.JS_EVENT_AXIS:

                        if number == self.ax_left_x:
                            self.vy = v

                        elif number == self.ax_left_y:
                            self.vx = -v

                        elif number == self.ax_right_x:
                            self.wz = v

        except Exception as e:
            print(f"[Joystick] Error: {e}")
            self._running = False



    def planar_local_velocity_reference(self, vx, vy, w, base_quat):
        """
        Compute a local-frame velocity reference such that the
        commanded motion remains parallel to the world horizontal plane.

        Parameters
        ----------
        linear_x : float
            Desired forward velocity [m/s]
        linear_y : float
            Desired lateral velocity [m/s]
        yaw_rate : float
            Desired yaw velocity [rad/s]
        base_quat : array-like, shape (4,)
            Base orientation quaternion [x, y, z, w]

        Returns
        -------
        v_local_ref : np.ndarray, shape (3,)
            Linear velocity reference expressed in the FULL local body frame.
            Its world projection is guaranteed to remain horizontal.

        yaw_rate : float
            Unchanged yaw rate reference.
        """

        # ---------------------------------------------------------
        # Full body rotation matrix
        # ---------------------------------------------------------

        R_wb = Rotation.from_quat(base_quat).as_matrix()

        # ---------------------------------------------------------
        # Build yaw-only heading frame
        # ---------------------------------------------------------

        # Body x-axis expressed in world frame
        x_body_world = R_wb[:, 0]

        # Project onto horizontal plane
        x_heading = x_body_world.copy()
        x_heading[2] = 0.0

        norm = np.linalg.norm(x_heading)

        if norm < 1e-6:
            x_heading = np.array([1.0, 0.0, 0.0])
        else:
            x_heading /= norm

        # World up axis
        z_world = np.array([0.0, 0.0, 1.0])

        # Heading y-axis
        y_heading = np.cross(z_world, x_heading)
        y_heading /= np.linalg.norm(y_heading)

        # Yaw-only rotation matrix
        R_wh = np.column_stack((x_heading, y_heading, z_world))

        # ---------------------------------------------------------
        # Desired planar velocity in heading frame
        # ---------------------------------------------------------

        v_heading = np.array([vx, vy, 0.0])

        # ---------------------------------------------------------
        # Convert to world frame (guaranteed planar)
        # ---------------------------------------------------------

        v_world = R_wh @ v_heading

        # ---------------------------------------------------------
        # Convert back to FULL body local frame
        # ---------------------------------------------------------

        v_local_ref = R_wb.T @ v_world

        return v_local_ref[0], v_local_ref[1], v_local_ref[2], w
