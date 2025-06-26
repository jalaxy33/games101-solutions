import numpy as np
import taichi as ti

from common import NpArr
import transform
from camera import Camera


@ti.data_oriented
class Renderer:
    def __init__(
        self, width: int, height: int, line_color=(1, 1, 1), background_color=(0, 0, 0)
    ):
        self.W = width
        self.H = height
        self.aspect_ratio = self.W / self.H

        self.line_color = ti.Vector(line_color, ti.float32)
        self.background_color = ti.Vector(background_color, ti.float32)

        self.image_buf = ti.Vector.field(3, ti.float32, shape=(self.W, self.H))

    def get_frame_array(self, cv2=True) -> NpArr:
        frame_array = self.image_buf.to_numpy()
        if cv2:
            frame_array = np.flip(frame_array.transpose(1, 0, 2), axis=0)
        return frame_array

    def set_camera(self, camera: Camera):
        self.camera = camera

        eye_pos = np.array(self.camera.eye_pos)
        vup = np.array(self.camera.vup)
        look_at = np.array(self.camera.look_at)
        vfov = self.camera.vfov
        zNear = self.camera.zNear
        zFar = self.camera.zFar

        self.viewport = transform.get_viewport_transform(self.W, self.H)
        self.view = transform.get_view_transform(eye_pos, look_at, vup)
        self.model = transform.get_model_transform()
        self.orthographic = transform.get_orthographic_transform(
            vfov, self.aspect_ratio, zNear, zFar
        )
        self.projection = transform.get_projection_transform(
            vfov, self.aspect_ratio, zNear, zFar
        )
        self.mvp = transform.get_mvp_transform(
            self.viewport, self.projection, self.view, self.model
        )

        # print("viewport:\n", self.viewport)
        # print("view:\n", self.view)
        # print("orthographic:\n", self.orthographic)
        # print("projection:\n", self.projection)
        # print("model:\n", self.model)
        # print("mvp:\n", self.mvp)

    def update_mvp_transform(self):
        self.mvp = transform.get_mvp_transform(
            self.viewport, self.projection, self.view, self.model
        )

    def set_mesh(self, vertices: NpArr, indices: NpArr):
        self.vertices = vertices.astype(np.float32)
        self.indices = indices.astype(np.int32)

        self.vertices_4d = np.hstack(
            [vertices, np.ones((len(vertices), 1))], dtype=np.float32
        )

        self.verts_gpu = ti.Vector.field(4, ti.float32)
        self.inds_gpu = ti.Vector.field(3, ti.int32)

        ti.root.dense(ti.i, len(vertices)).place(self.verts_gpu)
        ti.root.dense(ti.i, len(indices)).place(self.inds_gpu)

        self.verts_gpu.from_numpy(self.vertices_4d)
        self.inds_gpu.from_numpy(self.indices)

    def render(self):
        verts = self.vertices_4d @ self.mvp.T
        self.verts_gpu.from_numpy(verts)

        self.gpu_render()

    @ti.kernel
    def gpu_render(self):
        self.image_buf.fill(self.background_color)

        for i in self.inds_gpu:
            i1, i2, i3 = self.inds_gpu[i]
            v1, v2, v3 = self.verts_gpu[i1], self.verts_gpu[i2], self.verts_gpu[i3]

            v1 /= v1.w
            v2 /= v2.w
            v3 /= v3.w

            # self.set_pixel(v1.x, v1.y, self.line_color)
            # self.set_pixel(v2.x, v2.y, self.line_color)
            # self.set_pixel(v2.x, v2.y, self.line_color)

            self.draw_triangle_wireframe(v1, v2, v3)

    @ti.func
    def draw_triangle_wireframe(
        self, v1: ti.template(), v2: ti.template(), v3: ti.template()
    ):
        self.draw_line(v1, v2)
        self.draw_line(v2, v3)
        self.draw_line(v3, v1)

    @ti.func
    def draw_line(self, p1: ti.template(), p2: ti.template()):
        """
        Bresenham's line drawing algorithm.\n
        reference: https://github.com/miloyip/line/blob/master/line_bresenham.c
        """
        x1, y1 = int(p1.x), int(p1.y)
        x2, y2 = int(p2.x), int(p2.y)

        dx, dy = abs(x2 - x1), abs(y2 - y1)
        sx = int(ti.math.sign(x2 - x1 + 1e-12))  # sx = 1 if x0 < x1 else -1
        sy = int(ti.math.sign(y2 - y1 + 1e-12))  # sy = 1 if y0 < y1 else -1
        err = dx / 2
        if dx <= dy:
            err = -dy / 2

        x, y = x1, y1
        while x != x2 or y != y2:
            self.set_pixel(x, y, self.line_color)
            e2 = err
            if e2 > -dx:
                err -= dy
                x += sx
            if e2 < dy:
                err += dx
                y += sy

    @ti.func
    def set_pixel(self, x: int, y: int, color: ti.template()):
        if 0 <= x < self.W and 0 <= y < self.H:
            self.image_buf[x, y] = color
