import numpy as np
import taichi as ti

import transform
from camera import Camera
from common import NpArr, INF, Vec3
from triangle import Triangle


@ti.data_oriented
class Renderer:
    def __init__(self, width: int, height: int, background_color=(0, 0, 0)):
        self.W = width
        self.H = height
        self.aspect_ratio = self.W / self.H

        self.background_color = ti.Vector(background_color, ti.float32)

        self.image_buf = ti.Vector.field(3, ti.float32, shape=(self.W, self.H))
        self.depth_buf = ti.field(ti.float32, shape=(self.W, self.H))

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

    def set_mesh(self, vertices: NpArr, indices: NpArr, vertex_colors: NpArr):
        self.vertices = vertices.astype(np.float32)
        self.indices = indices.astype(np.int32)
        self.vertex_colors = vertex_colors.astype(np.float32)

        self.vertices_4d = np.hstack(
            [vertices, np.ones((len(vertices), 1))], dtype=np.float32
        )

        self.verts_gpu = ti.Vector.field(4, ti.float32)
        self.inds_gpu = ti.Vector.field(3, ti.int32)
        self.colors_gpu = ti.Vector.field(3, ti.float32)

        ti.root.dense(ti.i, len(vertices)).place(self.verts_gpu)
        ti.root.dense(ti.i, len(indices)).place(self.inds_gpu)
        ti.root.dense(ti.i, len(vertex_colors)).place(self.colors_gpu)

        self.verts_gpu.from_numpy(self.vertices_4d)
        self.inds_gpu.from_numpy(self.indices)
        self.colors_gpu.from_numpy(self.vertex_colors)

    def set_MSAA(self, n: int = 2):
        self.MSAA_N = n

    def render(self, use_MSAA: bool = False):
        verts = self.vertices_4d @ self.mvp.T
        self.verts_gpu.from_numpy(verts)

        self.use_MSAA = use_MSAA
        self.gpu_render(use_MSAA, self.MSAA_N)

    @ti.kernel
    def gpu_render(self, use_MSAA: bool, MSAA_N: int):
        self.image_buf.fill(self.background_color)
        self.depth_buf.fill(-INF)

        for i in self.inds_gpu:
            i1, i2, i3 = self.inds_gpu[i]
            v1, v2, v3 = self.verts_gpu[i1], self.verts_gpu[i2], self.verts_gpu[i3]
            c1, c2, c3 = self.colors_gpu[i1], self.colors_gpu[i2], self.colors_gpu[i3]

            v1 /= v1.w
            v2 /= v2.w
            v3 /= v3.w

            t = Triangle(v1, v2, v3, c1, c2, c3)

            if use_MSAA:
                self.draw_MSAA_triangle(t, MSAA_N)
            else:
                self.draw_triangle(t)

    # TODO: Find out the bounding box of current triangle.
    # iterate through the pixel and find if the current pixel is inside the triangle
    #
    # If so, use the following code to get the interpolated z value.
    # auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    # float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    # float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    # z_interpolated *= w_reciprocal;
    #
    # TODO: set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
    @ti.func
    def draw_triangle(self, t: Triangle):
        lb, ub = t.get_bounds()
        x_min, y_min = int(ti.floor(lb.x)), int(ti.floor(lb.y))
        x_max, y_max = int(ti.ceil(ub.x)), int(ti.ceil(ub.y))

        for x, y in ti.ndrange((x_min, x_max + 1), (y_min, y_max + 1)):
            p = Vec3(x, y, 1)
            if t.inside(p):
                z_interp, color_interp = t.interpolate_z_and_color(p)

                if self.depth_buf[x, y] < z_interp:
                    self.depth_buf[x, y] = z_interp
                    color = color_interp
                    self.set_pixel(x, y, color)

    # TODO: Implement MSAA version
    @ti.func
    def draw_MSAA_triangle(self, t: Triangle, MSAA_N: int):
        lb, ub = t.get_bounds()
        x_min, y_min = int(ti.floor(lb.x)), int(ti.floor(lb.y))
        x_max, y_max = int(ti.ceil(ub.x)), int(ti.ceil(ub.y))

        D = 1.0 / (2 * MSAA_N)
        MSAA_N2 = MSAA_N * MSAA_N
        for x, y in ti.ndrange((x_min, x_max + 1), (y_min, y_max + 1)):
            color = Vec3(0)
            depth = -INF
            for i, j in ti.ndrange(MSAA_N, MSAA_N):
                x_cur = x + 2 * D * i + D
                y_cur = y + 2 * D * j + D
                p = Vec3(x_cur, y_cur, 1.0)

                if t.inside(p):
                    z_interp, color_interp = t.interpolate_z_and_color(p)

                    color += color_interp
                    if depth < z_interp:
                        depth = z_interp

            if self.depth_buf[x, y] < depth:
                self.depth_buf[x, y] = depth
                color /= MSAA_N2
                self.set_pixel(x, y, color)

    @ti.func
    def set_pixel(self, x: int, y: int, color: ti.template()):
        if 0 <= x < self.W and 0 <= y < self.H:
            self.image_buf[x, y] = color
