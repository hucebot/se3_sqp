import glfw
import mujoco

class MujocoViewer:
    def __init__(
        self,
        model,
        width=1200,
        height=900,
        title="MuJoCo",
        maxgeom=1000,
    ):
        self.model = model

        # -----------------------------
        # GLFW init
        # -----------------------------
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        self.window = glfw.create_window(
            width,
            height,
            title,
            None,
            None,
        )

        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)

        # -----------------------------
        # MuJoCo visualization objects
        # -----------------------------
        self.context = mujoco.MjrContext(
            self.model,
            mujoco.mjtFontScale.mjFONTSCALE_150,
        )

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

        self.scene = mujoco.MjvScene(
            self.model,
            maxgeom=maxgeom,
        )

        # -----------------------------
        # Initialize free camera
        # -----------------------------
        mujoco.mjv_defaultFreeCamera(
            self.model,
            self.cam,
        )

        # -----------------------------
        # Mouse state
        # -----------------------------
        self.button_left = False
        self.button_middle = False
        self.button_right = False

        self.lastx = 0.0
        self.lasty = 0.0

        # -----------------------------
        # Register callbacks
        # -----------------------------
        glfw.set_cursor_pos_callback(
            self.window,
            self.mouse_move,
        )

        glfw.set_mouse_button_callback(
            self.window,
            self.mouse_button,
        )

        glfw.set_scroll_callback(
            self.window,
            self.scroll,
        )

    # =========================================================
    # Mouse callbacks
    # =========================================================

    def mouse_button(self, window, button, act, mods):
        self.button_left = (
            glfw.get_mouse_button(
                window,
                glfw.MOUSE_BUTTON_LEFT,
            )
            == glfw.PRESS
        )

        self.button_middle = (
            glfw.get_mouse_button(
                window,
                glfw.MOUSE_BUTTON_MIDDLE,
            )
            == glfw.PRESS
        )

        self.button_right = (
            glfw.get_mouse_button(
                window,
                glfw.MOUSE_BUTTON_RIGHT,
            )
            == glfw.PRESS
        )

    def mouse_move(self, window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty

        self.lastx = xpos
        self.lasty = ypos

        if not (
            self.button_left
            or self.button_middle
            or self.button_right
        ):
            return

        _, height = glfw.get_window_size(window)

        if self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H

        elif self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H

        elif self.button_middle:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        else:
            return

        mujoco.mjv_moveCamera(
            self.model,
            action,
            dx / height,
            dy / height,
            self.scene,
            self.cam,
        )

    def scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.scene,
            self.cam,
        )

    # =========================================================
    # Rendering
    # =========================================================

    def render(self, data):
        viewport = mujoco.MjrRect(
            0,
            0,
            *glfw.get_framebuffer_size(self.window),
        )

        mujoco.mjv_updateScene(
            self.model,
            data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )

        mujoco.mjr_render(
            viewport,
            self.scene,
            self.context,
        )

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    # =========================================================
    # Utilities
    # =========================================================

    def should_close(self):
        return glfw.window_should_close(self.window)

    def close(self):
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
