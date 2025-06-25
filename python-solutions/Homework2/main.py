# Homework 2. Draw 2 overlapping triangles & Implement Anti-aliasing.
import os

os.chdir(os.path.dirname(__file__))

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

# local packages
from common import *
from transform import *


@ti.pyfunc
def get_triangle_bbox(v1, v2, v3):
    lbound = ti.min(v1, v2, v3)
    ubound = ti.max(v1, v2, v3)
    return lbound, ubound


@ti.pyfunc
def inside_triangle(x: float, y: float, a: Vec3, b: Vec3, c: Vec3) -> bool:
    # TODO: Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    # alpha, beta, gamma = compute_barycentric(x, y, x1, y1, x2, y2, x3, y3)
    # return alpha >= 0 and beta >= 0 and gamma >= 0  # 在内部或边界上
    p = Vec3(x, y, 1)
    f1 = (p - a).cross(b - a).z > 0
    f2 = (p - b).cross(c - b).z > 0
    f3 = (p - c).cross(a - c).z > 0
    return (f1 == f2) and (f2 == f3)


@ti.pyfunc
def barycentric_ij(x: float, y: float, xi: float, yi: float, xj: float, yj: float) -> float:
    return (yi - yj) * x + (xj - xi) * y + xi * yj - xj * yi


# 计算三角形重心坐标
@ti.pyfunc
def compute_barycentric(x: float, y: float, a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    x1, y1 = a.x, a.y
    x2, y2 = b.x, b.y
    x3, y3 = c.x, c.y

    alpha = barycentric_ij(x, y, x2, y2, x3, y3) / barycentric_ij(x1, y1, x2, y2, x3, y3)
    beta = barycentric_ij(x, y, x3, y3, x1, y1) / barycentric_ij(x2, y2, x3, y3, x1, y1)
    # gamma = barycentric_ij(x, y, x1, y1, x2, y2) / barycentric_ij(
    #     x3, y3, x1, y1, x2, y2
    # )
    gamma = 1 - alpha - beta
    return Vec3(alpha, beta, gamma)


@ti.data_oriented
class Renderer:
    def __init__(self, width: int, height: int, background_color=(0, 0, 0), MSAA: bool = False, MSAA_N: int = 2):
        self.W = int(width)
        self.H = int(height)
        self.resolution = (self.W, self.H)
        self.aspect_ratio = 1.0 * height / width

        self.background_color = Vec3(background_color)

        self.MSAA = ti.field(int, shape=())
        self.MSAA_N = ti.field(int, shape=())
        self.MSAA.fill(MSAA)
        self.MSAA_N.fill(MSAA_N)

        self.frame_buf = ti.Vector.field(3, float, shape=self.resolution)  # 屏幕像素颜色信息（rbg）
        self.depth_buf = ti.field(float, shape=self.resolution)  # 屏幕像素深度信息（z-buffer）

        # transform matrices
        self.viewport = self.init_Mat4x4_field()
        self.view = self.init_Mat4x4_field()
        self.projection = self.init_Mat4x4_field()
        self.model = self.init_Mat4x4_field()
        self.mvp = self.init_Mat4x4_field()

    def init_Mat4x4_field(self):
        matrix = Mat4x4.field(shape=())
        matrix.fill(eye4())
        return matrix

    def set_camera(self, eye_pos: Vec3, vup: Vec3, vfov: float, look_at: Vec3, zNear: float, zFar: float):
        eye_pos = Vec3(eye_pos)
        vup = Vec3(vup)
        look_at = Vec3(look_at)

        self.viewport.fill(get_viewport_matrix(self.W, self.H))
        self.view.fill(get_view_matrix(eye_pos, look_at, vup))
        self.projection.fill(get_projection_matrix(vfov, self.aspect_ratio, zNear, zFar))
        self.model.fill(get_model_matrix())
        self.mvp.fill(self.viewport[None] @ self.projection[None] @ self.view[None] @ self.model[None])

        # print("viewport:\n", self.viewport)
        # print("view:\n", self.view)
        # print("projection:\n", self.projection)
        # print("model:\n", self.model)
        # print("mvp:\n", self.mvp)

    def set_mesh(self, vertices: NpArr, indices: NpArr, vertex_colors: NpArr):
        self.vertices = ti.Vector.field(3, float)
        self.indices = ti.Vector.field(3, int)
        self.vertex_colors = ti.Vector.field(3, float)

        ti.root.dense(ti.i, len(vertices)).place(self.vertices)
        ti.root.dense(ti.i, len(indices)).place(self.indices)
        ti.root.dense(ti.i, len(vertex_colors)).place(self.vertex_colors)

        self.vertices.from_numpy(vertices)
        self.indices.from_numpy(indices)
        self.vertex_colors.from_numpy(vertex_colors)

    @ti.kernel
    def render(self):
        self.frame_buf.fill(self.background_color)
        self.depth_buf.fill(-INF)

        for i in self.indices:
            i1, i2, i3 = self.indices[i]
            v1, v2, v3 = (
                self.mvp[None] @ Vec4(self.vertices[i1], 1),
                self.mvp[None] @ Vec4(self.vertices[i2], 1),
                self.mvp[None] @ Vec4(self.vertices[i3], 1),
            )
            c1, c2, c3 = self.vertex_colors[i1], self.vertex_colors[i2], self.vertex_colors[i3]

            v1 /= v1.w
            v2 /= v2.w
            v3 /= v3.w

            # self.set_pixel(v1.x, v1.y, c1)
            # self.set_pixel(v2.x, v2.y, c2)
            # self.set_pixel(v3.x, v3.y, c3)

            if self.MSAA[None]:
                self.rasterize_triangle_MSAA(v1, v2, v3, c1, c2, c3)
            else:
                self.rasterize_triangle(v1, v2, v3, c1, c2, c3)

    @ti.func
    def set_pixel(self, x: int, y: int, color: ti.template()):
        if 0 <= x < self.W and 0 <= y < self.H:
            self.frame_buf[x, y] = color

    @ti.func
    def rasterize_triangle(self, v1, v2, v3, c1, c2, c3):
        # TODO: Find out the bounding box of current triangle.
        # iterate through the pixel and find if the current pixel is inside the triangle

        # If so, use the following code to get the interpolated z value.
        # auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
        # float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
        # float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
        # z_interpolated *= w_reciprocal;

        # TODO: set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.

        lbound, ubound = get_triangle_bbox(v1, v2, v3)
        x_min, x_max = int(ti.floor(lbound.x)), int(ti.ceil(ubound.x))
        y_min, y_max = int(ti.floor(lbound.y)), int(ti.ceil(ubound.y))

        x1, y1, z1, w1 = v1
        x2, y2, z2, w2 = v2
        x3, y3, z3, w3 = v3

        for x, y in ti.ndrange((x_min, x_max + 1), (y_min, y_max + 1)):
            if inside_triangle(x, y, v1.xyz, v2.xyz, v3.xyz):
                alpha, beta, gamma = compute_barycentric(x, y, v1.xyz, v2.xyz, v3.xyz)
                w_reciprocal = 1.0 / (alpha / w1 + beta / w2 + gamma / w3)
                z_interpolated = alpha * z1 / w1 + beta * z2 / w2 + gamma * z3 / w3
                z_interpolated *= w_reciprocal

                if self.depth_buf[x, y] < z_interpolated:
                    self.depth_buf[x, y] = z_interpolated
                    color = alpha * c1 + beta * c2 + gamma * c3
                    self.set_pixel(x, y, color)

    @ti.func
    def rasterize_triangle_MSAA(self, v1, v2, v3, c1, c2, c3):
        # TODO: Implement MSAA version

        lbound, ubound = get_triangle_bbox(v1, v2, v3)
        x_min, x_max = int(ti.floor(lbound.x)), int(ti.ceil(ubound.x))
        y_min, y_max = int(ti.floor(lbound.y)), int(ti.ceil(ubound.y))

        x1, y1, z1, w1 = v1
        x2, y2, z2, w2 = v2
        x3, y3, z3, w3 = v3

        D = 1.0 / (2 * self.MSAA_N[None])
        for x, y in ti.ndrange((x_min, x_max + 1), (y_min, y_max + 1)):
            color = Vec3(0)
            depth = -INF
            for i, j in ti.ndrange(self.MSAA_N[None], self.MSAA_N[None]):
                x_curr = x + 2 * D * i + D
                y_curr = y + 2 * D * j + D

                if inside_triangle(x_curr, y_curr, v1.xyz, v2.xyz, v3.xyz):
                    alpha, beta, gamma = compute_barycentric(x_curr, y_curr, v1.xyz, v2.xyz, v3.xyz)
                    w_reciprocal = 1.0 / (alpha / w1 + beta / w2 + gamma / w3)
                    z_interpolated = alpha * z1 / w1 + beta * z2 / w2 + gamma * z3 / w3
                    z_interpolated *= w_reciprocal

                    c_interp = alpha * c1 + beta * c2 + gamma * c3
                    color += c_interp

                    if depth < z_interpolated:
                        depth = z_interpolated

            if self.depth_buf[x, y] < depth:
                self.depth_buf[x, y] = depth

                color /= self.MSAA_N[None] * self.MSAA_N[None]
                self.set_pixel(x, y, color)


if __name__ == "__main__":
    # define data
    vertices = np.array(
        [
            (2, 0, -2),
            (0, 2, -2),
            (-2, 0, -2),
            (3.5, -1, -5),
            (2.5, 1.5, -5),
            (-1, 0.5, -5),
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            (0, 1, 2),
            (3, 4, 5),
        ],
        dtype=np.int32,
    )
    vertex_colors = np.array(
        [
            (217.0, 238.0, 185.0),
            (217.0, 238.0, 185.0),
            (217.0, 238.0, 185.0),
            (185.0, 217.0, 238.0),
            (185.0, 217.0, 238.0),
            (185.0, 217.0, 238.0),
        ],
        dtype=np.float32,
    )
    vertex_colors = vertex_colors / 255.0

    assert len(vertices) == len(vertex_colors)

    # create renderer
    renderer = Renderer(width=1024, height=1024, background_color=(0, 0, 0), MSAA=True, MSAA_N=4)
    renderer.set_camera(
        eye_pos=(0, 0, 5),
        vup=(0, 1, 0),
        look_at=(0, 0, -5),
        vfov=60,
        zNear=0.1,
        zFar=50,
    )
    renderer.set_mesh(vertices, indices, vertex_colors)

    # rendering
    window = ti.ui.Window("Draw two triangles", renderer.resolution)
    canvas = window.get_canvas()

    renderer.render()

    hold_count = 0
    max_hold = 10
    while window.running and not window.is_pressed(ti.ui.ESCAPE):
        switch_flag = False
        if window.is_pressed(ti.ui.SPACE):  # 空格切换 Anti-Aliasing
            hold_count += 1
            if hold_count >= max_hold:  # 按住空格持续几帧才切换（防止抖动）
                renderer.MSAA[None] = not renderer.MSAA[None]
                switch_flag = True
                hold_count = 0

        if switch_flag:
            renderer.render()

        canvas.set_image(renderer.frame_buf)
        window.show()
