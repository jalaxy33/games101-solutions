import os

os.chdir(os.path.dirname(__file__))

import numpy as np
import taichi as ti
import taichi.math as tm
import trimesh
import skimage

# local packages
from transform import *


ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

# define types
vec2 = ti.types.vector(2, float)
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
        normals: np.ndarray,
        vertex_colors: np.ndarray,
        texture_uvs: np.ndarray,
    ):
        self.verticies = ti.Vector.field(3, float, shape=len(vertices))
        self.indicies = ti.Vector.field(3, int, shape=len(indicies))
        self.normals = ti.Vector.field(3, float, shape=len(normals))
        self.vertex_colors = ti.Vector.field(3, float, shape=len(vertex_colors))
        self.texture_uvs = ti.Vector.field(2, float, shape=len(texture_uvs))

        self.verticies.from_numpy(vertices.astype(np.float32))
        self.indicies.from_numpy(indicies.astype(np.int32))
        self.normals.from_numpy(normals.astype(np.float32))
        self.vertex_colors.from_numpy(vertex_colors.astype(np.float32))
        self.texture_uvs.from_numpy(texture_uvs.astype(np.float32))


@ti.data_oriented
class Texture:
    def __init__(self, image: np.ndarray, name=""):
        self.name = name
        self.W = image.shape[0]
        self.H = image.shape[1]
        self.image = ti.Vector.field(3, float, shape=(self.W, self.H))
        self.image.from_numpy(image)

    def get_color(self, u: float, v: float):
        i, j = int(u * self.W), int(v * self.H)
        color = vec3(0)
        if i >= 0 and i < self.W and j >= 0 and j < self.H:
            color = self.image[i, j]
        return color


@ti.dataclass
class Triangle:
    # 顶点坐标
    a: vec3
    b: vec3
    c: vec3
    # 顶点法向
    n1: vec3
    n2: vec3
    n3: vec3
    # 顶点颜色
    c1: vec3
    c2: vec3
    c3: vec3
    # 材质坐标
    uv1: vec2
    uv2: vec2
    uv3: vec2
    # 相机空间坐标
    vp1: vec3
    vp2: vec3
    vp3: vec3

    @ti.func
    def set_vertices(self, a: vec3, b: vec3, c: vec3):
        self.a = a
        self.b = b
        self.c = c

    @ti.func
    def set_normals(self, n1: vec3, n2: vec3, n3: vec3):
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

    @ti.func
    def set_colors(self, c1: vec3, c2: vec3, c3: vec3):
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    @ti.func
    def set_texture_uvs(self, uv1: vec2, uv2: vec2, uv3: vec2):
        self.uv1 = uv1
        self.uv2 = uv2
        self.uv3 = uv3


@ti.dataclass
class FragmentShaderPayload:
    x: float
    y: float
    color: vec3
    normal: vec3
    viewspace_pos: vec3
    texture_uv: vec2


@ti.data_oriented
class Rasterizer:
    def __init__(
        self,
        width: float,
        height: float,
        background_color=(0, 0, 0),
        MSAA: bool = False,
        MSAA_N: int = 2,
        data_angles=(0, 0, 0),
        data_scales=(1, 1, 1),
        data_translates=(0, 0, 0),
        shader_option: int = 0,
    ):
        self.W = int(width)
        self.H = int(height)
        self.resolution = (self.W, self.H)
        self.aspect_ratio = 1.0 * height / width

        self.background_color = vec3(background_color)

        self.MSAA = ti.field(int, shape=())
        self.MSAA[None] = MSAA
        self.MSAA_N = MSAA_N

        self.data_angles = data_angles
        self.data_scales = data_scales
        self.data_translates = data_translates

        self.frame_buf = ti.Vector.field(3, float, shape=self.resolution)  # 屏幕像素颜色信息（rbg）
        self.depth_buf = ti.field(float, shape=self.resolution)  # 屏幕像素深度信息（z-buffer）

        # 变换矩阵
        self.model = mat4(np.eye(4))
        self.view = mat4(np.eye(4))
        self.projection = mat4(np.eye(4))
        self.viewport = mat4(np.eye(4))

        self.mvp = mat4(np.eye(4))
        self.model_view = mat4(np.eye(4))
        self.model_view_inv = mat4(np.eye(4))

        # 着色器选项
        self.shader_option = ti.field(int, shape=())
        self.shader_option[None] = shader_option

    def set_camera(self, camera: Camera):
        self.camera = camera

        self.model = get_model_matrix(self.data_angles, self.data_scales, self.data_translates)
        self.view = get_view_matrix(camera.pos, camera.look_at, camera.up)
        self.projection = get_projection_matrix(camera.fov, self.aspect_ratio, camera.zNear, camera.zFar)
        self.viewport = get_viewport_matrix(self.W, self.H)

        self.mvp = self.compute_mvp()
        self.model_view = self.compute_model_view()
        self.model_view_inv = self.compute_model_view_inv()

    def set_mesh(self, mesh: Mesh):
        self.mesh = mesh

    def set_texture(self, texture: Texture):
        self.texture = texture

    def compute_mvp(self) -> mat4:
        return self.projection @ self.view @ self.model

    def compute_model_view(self) -> mat4:
        return self.view @ self.model

    def compute_model_view_inv(self) -> mat4:
        return self.model_view.inverse().transpose()

    @ti.func
    def set_pixel(self, x: int, y: int, color: vec3):
        if x >= 0 and x < self.W and y >= 0 and y < self.H:
            self.frame_buf[x, y] = color

    @ti.func
    def default_fragment_shader(self, payload: FragmentShaderPayload):
        x, y = payload.x, payload.y
        color = payload.color
        self.set_pixel(x, y, color)

    @ti.func
    def rasterize_triangle(self, t: Triangle):
        # TODO: From your HW2, get the triangle rasterization code.
        # TODO: Inside your rasterization loop:
        #    * v[i].w() is the vertex view space depth value z.
        #    * Z is interpolated view space depth for the current pixel
        #    * zp is depth between zNear and zFar, used for z-buffer

        # float Z = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
        # float zp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
        # zp *= Z;

        # TODO: Interpolate the attributes:
        # auto interpolated_color
        # auto interpolated_normal
        # auto interpolated_texcoords
        # auto interpolated_shadingcoords

        # Use: fragment_shader_payload payload( interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
        # Use: payload.view_pos = interpolated_shadingcoords;
        # Use: Instead of passing the triangle's color directly to the frame buffer, pass the color to the shaders first to get the final color;
        # Use: auto pixel_color = fragment_shader(payload);
        v1, v2, v3 = vec4(t.a, 1), vec4(t.b, 1), vec4(t.c, 1)
        n1, n2, n3 = vec4(t.n1, 0), vec4(t.n2, 0), vec4(t.n3, 0)
        c1, c2, c3 = t.c1, t.c2, t.c3
        uv1, uv2, uv3 = t.uv1, t.uv2, t.uv3

        n1, n2, n3 = (
            self.model_view_inv @ n1,
            self.model_view_inv @ n2,
            self.model_view_inv @ n3,
        )

        # viewspace positions
        vp1, vp2, vp3 = (
            (self.model_view @ v1).xyz,
            (self.model_view @ v2).xyz,
            (self.model_view @ v3).xyz,
        )

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

                    color = (alpha * c1 + beta * c2 + gamma * c3).xyz
                    normal = (alpha * n1 + beta * n2 + gamma * n3).xyz
                    viewspace_pos = (alpha * vp1 + beta * vp2 + gamma * vp3).xyz
                    texture_uv = (alpha * uv1 + beta * uv2 + gamma * uv3).xy

                    payload = FragmentShaderPayload(x, y, color, normal, viewspace_pos, texture_uv)

                    # self.set_pixel(x, y, color)
                    if self.shader_option[None] == 0:
                        self.default_fragment_shader(payload)

    @ti.func
    def rasterize_triangle_MSAA(self, t: Triangle):
        # TODO: From your HW2, get the triangle rasterization code.
        # TODO: Inside your rasterization loop:
        #    * v[i].w() is the vertex view space depth value z.
        #    * Z is interpolated view space depth for the current pixel
        #    * zp is depth between zNear and zFar, used for z-buffer

        # float Z = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
        # float zp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
        # zp *= Z;

        # TODO: Interpolate the attributes:
        # auto interpolated_color
        # auto interpolated_normal
        # auto interpolated_texcoords
        # auto interpolated_shadingcoords

        # Use: fragment_shader_payload payload( interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
        # Use: payload.view_pos = interpolated_shadingcoords;
        # Use: Instead of passing the triangle's color directly to the frame buffer, pass the color to the shaders first to get the final color;
        # Use: auto pixel_color = fragment_shader(payload);
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
                    alpha, beta, gamma = compute_barycentric(x, y, v1.xyz, v2.xyz, v3.xyz)
                    w_reciprocal = 1.0 / (alpha / w1 + beta / w2 + gamma / w3)
                    z_interpolated = alpha * z1 / w1 + beta * z2 / w2 + gamma * z3 / w3
                    z_interpolated *= w_reciprocal

                    c_interp = alpha * c1 + beta * c2 + gamma * c3
                    color += c_interp

                    if depth < z_interpolated:
                        depth = z_interpolated

            color /= self.MSAA_N * self.MSAA_N
            if self.depth_buf[x, y] < depth:
                self.depth_buf[x, y] = depth
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
            n1, n2, n3 = (
                self.mesh.normals[i1],
                self.mesh.normals[i2],
                self.mesh.normals[i3],
            )
            c1, c2, c3 = (
                self.mesh.vertex_colors[i1],
                self.mesh.vertex_colors[i2],
                self.mesh.vertex_colors[i3],
            )
            uv1, uv2, uv3 = (
                self.mesh.texture_uvs[i1],
                self.mesh.texture_uvs[i2],
                self.mesh.texture_uvs[i3],
            )

            t = Triangle()
            t.set_vertices(a, b, c)
            t.set_normals(n1, n2, n3)
            t.set_colors(c1, c2, c3)
            t.set_texture_uvs(uv1, uv2, uv3)

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
    p = vec3(x, y, 1)
    f1 = tm.cross(p - a, b - a).z > 0
    f2 = tm.cross(p - b, c - b).z > 0
    f3 = tm.cross(p - c, a - c).z > 0
    return (f1 == f2) and (f2 == f3)


if __name__ == "__main__":
    # load model
    obj_file = os.path.abspath("./task/Code/models/spot/spot_triangulated_good.obj")
    texture_png1 = os.path.abspath("./task/Code/models/spot/spot_texture.png")
    texture_png2 = os.path.abspath("./task/Code/models/spot/hmap.jpg")
    mesh = trimesh.load_mesh(obj_file)
    tex_img1 = skimage.io.imread(texture_png1).astype(np.float32) / 255  # H, W, C
    tex_img2 = skimage.io.imread(texture_png2).astype(np.float32) / 255  # H, W, C

    verts = mesh.vertices.astype(np.float32)
    inds = mesh.faces.astype(np.int32)
    normals = mesh.vertex_normals.astype(np.float32)
    vert_colors = mesh.visual.to_color().vertex_colors.astype(np.float32) / 255
    tex_uvs = mesh.visual.uv.astype(np.float32)

    tex_img1 = tex_img1.transpose(1, 0, 2)[:, ::-1, :]
    tex_img2 = tex_img2.transpose(1, 0, 2)[:, ::-1, :]

    # prepare mesh and texture
    mesh = Mesh(verts, inds, normals, vert_colors, tex_uvs)
    texture1 = Texture(tex_img1)
    texture2 = Texture(tex_img2)

    # create camera
    camera = Camera(pos=(0, 0, 5), up=(0, 1, 0), look_at=(0, 0, -5), fov=60, zNear=0.1, zFar=50)

    # define display
    width = 1024
    height = 1024
    background_color = (0, 0, 0)

    MSAA = False
    MSAA_N = 2  # MSAA-NxN

    # data transform
    data_angles = (0, -140, 0)
    data_scales = vec3(2.5, 2.5, 2.5)
    data_translates = vec3(0, 0, 0)

    # shader
    shader_option = 0

    # create rasterizer
    rasterizer = Rasterizer(
        width,
        height,
        background_color,
        MSAA,
        MSAA_N,
        data_angles,
        data_scales,
        data_translates,
        shader_option,
    )
    rasterizer.set_camera(camera)
    rasterizer.set_mesh(mesh)
    rasterizer.set_texture(texture1)
    # rasterizer.set_texture(texture2)

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

        # 选择着色器
        if window.is_pressed(ti.ui.SPACE):
            rasterizer.shader_option[None] = 0
            change_flag = True

        if change_flag:
            rasterizer.render()

        canvas.set_image(rasterizer.frame_buf)
        window.show()
