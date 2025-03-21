# Assignment 2. Draw 2 overlapping triangles & Implement Anti-aliasing.
import os

os.chdir(os.path.dirname(__file__))

import numpy as np
import taichi as ti
import taichi.math as tm

# local packages
from transform import *


ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

# define types
vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)
mat4 = ti.types.matrix(4, 4, float)


@ti.dataclass
class Camera:
    pos: vec3
    up: vec3
    look_at: vec3
    fov: float
    zNear: float
    zFar: float


@ti.data_oriented
class Mesh:
    def __init__(
        self,
        vertices: np.ndarray,
        indicies: np.ndarray,
        vertex_colors: np.ndarray,
    ):
        self.verticies = ti.Vector.field(3, float, shape=len(vertices))
        self.indicies = ti.Vector.field(3, int, shape=len(indicies))
        self.vertex_colors = ti.Vector.field(3, float, shape=len(vertex_colors))

        self.verticies.from_numpy(vertices.astype(np.float32))
        self.indicies.from_numpy(indicies.astype(np.int32))
        self.vertex_colors.from_numpy(vertex_colors.astype(np.float32))


@ti.dataclass
class Triangle:
    # 顶点坐标
    a: vec3
    b: vec3
    c: vec3
    # 顶点颜色
    c1: vec3
    c2: vec3
    c3: vec3

    @ti.func
    def set_vertices(self, a: vec3, b: vec3, c: vec3):
        self.a = a
        self.b = b
        self.c = c

    @ti.func
    def set_colors(self, c1: vec3, c2: vec3, c3: vec3):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3


@ti.data_oriented
class Rasterizer:
    def __init__(self, width: float, height: float, background_color=(0, 0, 0), MSAA: bool = False, MSAA_N: int = 2):
        self.W = int(width)
        self.H = int(height)
        self.resolution = (self.W, self.H)
        self.aspect_ratio = 1.0 * height / width

        self.background_color = vec3(background_color)

        self.MSAA = ti.field(int, shape=())
        self.MSAA[None] = MSAA
        self.MSAA_N = MSAA_N

        self.frame_buf = ti.Vector.field(3, float, shape=self.resolution)  # 屏幕像素颜色信息（rbg）
        self.depth_buf = ti.field(float, shape=self.resolution)  # 屏幕像素深度信息（z-buffer）

        # 变换矩阵
        self.model = mat4(np.eye(4))
        self.view = mat4(np.eye(4))
        self.projection = mat4(np.eye(4))
        self.viewport = mat4(np.eye(4))
        self.mvp = mat4(np.eye(4))

    def set_camera(self, camera: Camera):
        self.camera = camera

        self.model = get_model_matrix()
        self.view = get_view_matrix(camera.pos, camera.look_at, camera.up)
        self.projection = get_projection_matrix(camera.fov, self.aspect_ratio, camera.zNear, camera.zFar)
        self.viewport = get_viewport_matrix(self.W, self.H)
        self.mvp = self.compute_mvp()

    def set_mesh(self, mesh: Mesh):
        self.mesh = mesh

    def compute_mvp(self) -> mat4:
        mvp = self.projection @ self.view @ self.model
        return mvp

    @ti.func
    def set_pixel(self, x: int, y: int, color: vec3):
        if x >= 0 and x < self.W and y >= 0 and y < self.H:
            self.frame_buf[x, y] = color

    @ti.func
    def rasterize_triangle(self, t: Triangle):
        # TODO: Find out the bounding box of current triangle.
        # iterate through the pixel and find if the current pixel is inside the triangle

        # If so, use the following code to get the interpolated z value.
        # auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
        # float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
        # float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
        # z_interpolated *= w_reciprocal;

        # TODO: set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.

        v1, v2, v3 = vec4(t.a, 1), vec4(t.b, 1), vec4(t.c, 1)
        c1, c2, c3 = t.c1, t.c2, t.c3

        # model-view-projection
        v1 = self.mvp @ v1
        v2 = self.mvp @ v2
        v3 = self.mvp @ v3

        # Homogeneous division
        v1 /= v1.w
        v2 /= v2.w
        v3 /= v3.w

        # Viewport transformation
        v1 = self.viewport @ v1
        v2 = self.viewport @ v2
        v3 = self.viewport @ v3

        v1.w = 1
        v2.w = 1
        v3.w = 1

        # 创建 bounding box
        x_min = int(tm.floor(tm.min(v1.x, v2.x, v3.x)))
        x_max = int(tm.ceil(tm.max(v1.x, v2.x, v3.x)))
        y_min = int(tm.floor(tm.min(v1.y, v2.y, v3.y)))
        y_max = int(tm.ceil(tm.max(v1.y, v2.y, v3.y)))

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
    def rasterize_triangle_MSAA(self, t: Triangle):
        # TODO: Find out the bounding box of current triangle.
        # iterate through the pixel and find if the current pixel is inside the triangle

        # If so, use the following code to get the interpolated z value.
        # auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
        # float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
        # float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
        # z_interpolated *= w_reciprocal;

        # TODO: set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.

        v1, v2, v3 = vec4(t.a, 1), vec4(t.b, 1), vec4(t.c, 1)
        c1, c2, c3 = t.c1, t.c2, t.c3

        # model-view-projection
        v1 = self.mvp @ v1
        v2 = self.mvp @ v2
        v3 = self.mvp @ v3

        # Homogeneous division
        v1 /= v1.w
        v2 /= v2.w
        v3 /= v3.w

        # Viewport transformation
        v1 = self.viewport @ v1
        v2 = self.viewport @ v2
        v3 = self.viewport @ v3

        v1.w = 1
        v2.w = 1
        v3.w = 1

        # 创建 bounding box
        x_min = int(tm.floor(tm.min(v1.x, v2.x, v3.x)))
        x_max = int(tm.ceil(tm.max(v1.x, v2.x, v3.x)))
        y_min = int(tm.floor(tm.min(v1.y, v2.y, v3.y)))
        y_max = int(tm.ceil(tm.max(v1.y, v2.y, v3.y)))

        x1, y1, z1, w1 = v1
        x2, y2, z2, w2 = v2
        x3, y3, z3, w3 = v3

        D = 1.0 / (2 * self.MSAA_N)
        for x, y in ti.ndrange((x_min, x_max + 1), (y_min, y_max + 1)):
            color = vec3(0)
            depth = -tm.inf
            for i, j in ti.ndrange(self.MSAA_N, self.MSAA_N):
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

                color /= self.MSAA_N * self.MSAA_N
                self.set_pixel(x, y, color)

    @ti.kernel
    def render(self):
        self.frame_buf.fill(self.background_color)
        self.depth_buf.fill(-tm.inf)

        for i in self.mesh.indicies:
            i1, i2, i3 = self.mesh.indicies[i]
            a, b, c = (
                self.mesh.verticies[i1],
                self.mesh.verticies[i2],
                self.mesh.verticies[i3],
            )
            c1, c2, c3 = (
                self.mesh.vertex_colors[i1],
                self.mesh.vertex_colors[i2],
                self.mesh.vertex_colors[i3],
            )

            t = Triangle()
            t.set_vertices(a, b, c)
            t.set_colors(c1, c2, c3)

            if self.MSAA[None]:
                self.rasterize_triangle_MSAA(t)
            else:
                self.rasterize_triangle(t)


@ti.func
def barycentric_ij(x: float, y: float, xi: float, yi: float, xj: float, yj: float):
    return (yi - yj) * x + (xj - xi) * y + xi * yj - xj * yi


# 计算三角形重心坐标
@ti.func
def compute_barycentric(x: float, y: float, a: vec3, b: vec3, c: vec3) -> vec3:
    x1, y1 = a.x, a.y
    x2, y2 = b.x, b.y
    x3, y3 = c.x, c.y

    alpha = barycentric_ij(x, y, x2, y2, x3, y3) / barycentric_ij(x1, y1, x2, y2, x3, y3)
    beta = barycentric_ij(x, y, x3, y3, x1, y1) / barycentric_ij(x2, y2, x3, y3, x1, y1)
    # gamma = barycentric_ij(x, y, x1, y1, x2, y2) / barycentric_ij(
    #     x3, y3, x1, y1, x2, y2
    # )
    gamma = 1 - alpha - beta
    return vec3(alpha, beta, gamma)


@ti.func
def inside_triangle(x: float, y: float, a: vec3, b: vec3, c: vec3) -> bool:
    # TODO: Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    # alpha, beta, gamma = compute_barycentric(x, y, x1, y1, x2, y2, x3, y3)
    # return alpha >= 0 and beta >= 0 and gamma >= 0  # 在内部或边界上
    p = vec3(x, y, 1)
    f1 = tm.cross(p - a, b - a).z > 0
    f2 = tm.cross(p - b, c - b).z > 0
    f3 = tm.cross(p - c, a - c).z > 0
    return (f1 == f2) and (f2 == f3)


if __name__ == "__main__":
    # define data
    pos = np.array(
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

    inds = np.array([(0, 1, 2), (3, 4, 5)], dtype=np.int32)

    cols = np.array(
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
    cols = cols / 255.0

    assert len(pos) == len(cols)

    # build mesh
    mesh = Mesh(vertices=pos, indicies=inds, vertex_colors=cols)

    # create camera
    camera = Camera(pos=(0, 0, 5), up=(0, 1, 0), look_at=(0, 0, -5), fov=60, zNear=0.1, zFar=50)

    # define display
    width = 1024
    height = 1024
    background_color = (0, 0, 0)

    MSAA = True
    MSAA_N = 4  # MSAA-NxN

    # create rasterizer
    rasterizer = Rasterizer(width, height, background_color, MSAA, MSAA_N)
    rasterizer.set_camera(camera)
    rasterizer.set_mesh(mesh)

    # print("viewport:\n", rasterizer.viewport)
    # print("projection:\n", rasterizer.projection)
    # print("view:\n", rasterizer.view)
    # print("model:\n", rasterizer.model)
    # print("mvp:\n", rasterizer.mvp)

    # rendering
    window = ti.ui.Window("draw two triangles", rasterizer.resolution)
    canvas = window.get_canvas()

    rasterizer.render()

    while window.running:
        if window.is_pressed(ti.ui.ESCAPE):  # 按ESC退出
            break

        change_flag = False
        if window.is_pressed(ti.ui.TAB):  # TAB切换 Anti-Aliasing
            rasterizer.MSAA[None] = not rasterizer.MSAA[None]
            change_flag = True

        if change_flag:
            rasterizer.render()

        canvas.set_image(rasterizer.frame_buf)
        window.show()
