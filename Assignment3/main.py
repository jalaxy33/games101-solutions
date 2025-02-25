# Assignment 3. Implement shaders

import os

os.chdir(os.path.dirname(__file__))

import copy
from typing import *
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
mat3 = ti.types.matrix(3, 3, float)
mat4 = ti.types.matrix(4, 4, float)


@ti.dataclass
class Camera:
    pos: vec3
    up: vec3
    look_at: vec3
    fov: float
    zNear: float
    zFar: float


@ti.dataclass
class Light:
    position: vec3
    intensity: vec3


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

    @ti.func
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
        self.model = mat4.field(shape=())
        self.view = mat4.field(shape=())
        self.projection = mat4.field(shape=())
        self.viewport = mat4.field(shape=())

        self.model[None] = mat4(np.eye(4))
        self.view[None] = mat4(np.eye(4))
        self.projection[None] = mat4(np.eye(4))
        self.viewport[None] = mat4(np.eye(4))

        self.mvp = mat4.field(shape=())
        self.model_view = mat4.field(shape=())
        self.model_view_inv = mat4.field(shape=())

        self.mvp[None] = mat4(np.eye(4))
        self.model_view[None] = mat4(np.eye(4))
        self.model_view_inv[None] = mat4(np.eye(4))

        # 片段着色器选项
        # （0-默认，1-法向贴图，2-Bling-Phong，3-纹理映射，4-位移贴图，5-凹凸贴图）
        self.frag_shader_option = ti.field(int, shape=())
        self.frag_shader_option[None] = shader_option

        # 光照参数
        self.kd = vec3(1, 1, 1)  # 漫反射系数
        self.ks = vec3(0.7937, 0.7937, 0.7937)  # 镜面反射系数
        self.ka = vec3(0.005, 0.005, 0.005)  # 环境光系数
        self.Ia = vec3(10, 10, 10)  # 环境光强度
        self.phong_exp = 150.0  # 高光区域集中度

        # bump mapping & displacement mapping
        self.kh = 0.2 * 15
        self.kn = 0.1 * 15

    def set_camera(self, camera: Camera):
        self.camera = camera

        self.model[None] = get_model_matrix(self.data_angles, self.data_scales, self.data_translates)
        self.view[None] = get_view_matrix(camera.pos, camera.look_at, camera.up)
        self.projection[None] = get_projection_matrix(
            camera.fov, self.aspect_ratio, camera.zNear, camera.zFar
        )
        self.viewport[None] = get_viewport_matrix(self.W, self.H)

        self.update_matrices()

    def set_mesh(self, mesh: Mesh):
        self.mesh = mesh

    def set_texture(self, texture: Texture):
        self.texture = texture

    def set_lights(self, *lights):
        self.lights = Light.field(shape=len(lights))
        for i in range(len(lights)):
            self.lights[i] = lights[i]

    def update_matrices(self):
        self.mvp[None] = self.compute_mvp()
        self.model_view[None] = self.compute_model_view()
        self.model_view_inv[None] = self.compute_model_view_inv()

    def compute_mvp(self) -> mat4:
        return self.projection[None] @ self.view[None] @ self.model[None]

    def compute_model_view(self) -> mat4:
        return self.view[None] @ self.model[None]

    def compute_model_view_inv(self) -> mat4:
        return self.model_view[None].inverse().transpose()

    @ti.func
    def set_pixel(self, x: int, y: int, color: vec3):
        if x >= 0 and x < self.W and y >= 0 and y < self.H:
            self.frame_buf[x, y] = color

    # 片段着色器：默认
    @ti.func
    def default_fragment_shader(self, payload: FragmentShaderPayload) -> vec3:
        color = payload.color
        return color

    # 片段着色器：法向贴图
    @ti.func
    def normal_fragment_shader(self, payload: FragmentShaderPayload) -> vec3:
        color = (tm.normalize(payload.normal) + vec3(1)) / 2
        return color

    # 片段着色器：Bling-Phong
    @ti.func
    def bling_phong_fragment_shader(self, payload: FragmentShaderPayload) -> vec3:
        pos = payload.viewspace_pos
        normal = tm.normalize(payload.normal)  # 表面法向
        eye_pos = self.camera.pos
        kd = payload.color  # 调整漫反射系数为颜色

        color = vec3(0)
        # for i in self.lights:
        for i in ti.ndrange(self.lights.shape[0]):
            # TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular*
            # components are. Then, accumulate that result on the *result_color* object.
            light = self.lights[i]
            light_pos = light.position.xyz
            light_intensity = light.intensity.xyz

            r = tm.distance(light_pos, pos)  # 光源-点距离
            light_dir = tm.normalize(light_pos - pos)  # 光源-点方向
            view_dir = tm.normalize(eye_pos - pos)  # 相机-点方向
            half_dir = tm.normalize(view_dir + light_dir)  # 半程向量

            E = light_intensity / (r * r)  # 此处的光能

            ambient = self.ka * self.Ia
            diffuse = kd * E * tm.max(0, tm.dot(normal, light_dir))
            specular = self.ks * E * tm.pow(tm.max(0, tm.dot(normal, half_dir)), self.phong_exp)

            color += ambient + diffuse + specular
        return color

    # 片段着色器：纹理映射
    @ti.func
    def texture_fragment_shader(self, payload: FragmentShaderPayload) -> vec3:
        # TODO: Get the texture value at the texture coordinates of the current fragment
        u, v = payload.texture_uv
        payload.color = self.texture.get_color(u, v)
        color = self.bling_phong_fragment_shader(payload)
        return color

    # 片段着色器：位移贴图
    @ti.func
    def displacement_fragment_shader(self, payload: FragmentShaderPayload) -> vec3:
        # TODO: Implement displacement mapping here
        # Let n = normal = (x, y, z)
        # Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
        # Vector b = n cross product t
        # Matrix TBN = [t b n]
        # dU = kh * kn * (h(u+1/w,v)-h(u,v))
        # dV = kh * kn * (h(u,v+1/h)-h(u,v))
        # Vector ln = (-dU, -dV, 1)
        # Position p = p + kn * n * h(u,v)
        # Normal n = normalize(TBN * ln)
        point = payload.viewspace_pos
        u, v = payload.texture_uv
        W, H = self.texture.W, self.texture.H

        n = tm.normalize(payload.normal)
        nx, ny, nz = n.xyz
        sqrt_xz = tm.sqrt(nx * nx + nz * nz)
        t = vec3(nx * ny / sqrt_xz, sqrt_xz, nz * ny / sqrt_xz)
        b = tm.cross(n, t)

        TBN = mat3(
            [
                [t.x, b.x, n.x],
                [t.y, b.y, n.y],
                [t.z, b.z, n.z],
            ]
        )

        dU = self.kh * self.kn
        dU *= self.texture.get_color(u + 1.0 / W, v).norm() - self.texture.get_color(u, v).norm()
        dV = self.kh * self.kn
        dV *= self.texture.get_color(u, v + 1.0 / H).norm() - self.texture.get_color(u, v).norm()
        ln = vec3(-dU, -dV, 1)

        payload.color = self.texture.get_color(u, v)
        payload.viewspace_pos = point + self.kn * n * self.texture.get_color(u, v).norm()
        payload.normal = tm.normalize(TBN @ ln)
        color = self.bling_phong_fragment_shader(payload)
        return color

    # 片段着色器：凹凸贴图
    @ti.func
    def bump_fragment_shader(self, payload: FragmentShaderPayload) -> vec3:
        # TODO: Implement bump mapping here
        # Let n = normal = (x, y, z)
        # Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
        # Vector b = n cross product t
        # Matrix TBN = [t b n]
        # dU = kh * kn * (h(u+1/w,v)-h(u,v))
        # dV = kh * kn * (h(u,v+1/h)-h(u,v))
        # Vector ln = (-dU, -dV, 1)
        # Position p = p + kn * n * h(u,v)
        # Normal n = normalize(TBN * ln)
        u, v = payload.texture_uv
        W, H = self.texture.W, self.texture.H

        n = tm.normalize(payload.normal)
        nx, ny, nz = n.xyz
        sqrt_xz = tm.sqrt(nx * nx + nz * nz)
        t = vec3(nx * ny / sqrt_xz, sqrt_xz, nz * ny / sqrt_xz)
        b = tm.cross(n, t)

        TBN = mat3(
            [
                [t.x, b.x, n.x],
                [t.y, b.y, n.y],
                [t.z, b.z, n.z],
            ]
        )

        dU = self.kh * self.kn
        dU *= self.texture.get_color(u + 1.0 / W, v).norm() - self.texture.get_color(u, v).norm()
        dV = self.kh * self.kn
        dV *= self.texture.get_color(u, v + 1.0 / H).norm() - self.texture.get_color(u, v).norm()
        ln = vec3(-dU, -dV, 1)

        color = tm.normalize(TBN @ ln)
        return color

    # 选择片段着色器
    @ti.func
    def fragment_shader(self, payload: FragmentShaderPayload) -> vec3:
        pixel_color = payload.color
        if self.frag_shader_option[None] == 1:
            # 法向贴图
            pixel_color = self.normal_fragment_shader(payload)
        elif self.frag_shader_option[None] == 2:
            # Bling-Phong
            pixel_color = self.bling_phong_fragment_shader(payload)
        elif self.frag_shader_option[None] == 3:
            # 纹理映射
            pixel_color = self.texture_fragment_shader(payload)
        elif self.frag_shader_option[None] == 4:
            # 位移贴图
            pixel_color = self.displacement_fragment_shader(payload)
        elif self.frag_shader_option[None] == 5:
            # 凹凸贴图
            pixel_color = self.bump_fragment_shader(payload)
        else:
            pixel_color = self.default_fragment_shader(payload)
        return pixel_color

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
            self.model_view_inv[None] @ n1,
            self.model_view_inv[None] @ n2,
            self.model_view_inv[None] @ n3,
        )

        # viewspace positions
        vp1, vp2, vp3 = (
            (self.model_view[None] @ v1).xyz,
            (self.model_view[None] @ v2).xyz,
            (self.model_view[None] @ v3).xyz,
        )

        # model-view-projection
        v1 = self.mvp[None] @ v1
        v2 = self.mvp[None] @ v2
        v3 = self.mvp[None] @ v3

        # Homogeneous division
        v1 /= v1.w
        v2 /= v2.w
        v3 /= v3.w

        # Viewport transformation
        v1 = self.viewport[None] @ v1
        v2 = self.viewport[None] @ v2
        v3 = self.viewport[None] @ v3

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

                    payload = FragmentShaderPayload(
                        x, y, color, tm.normalize(normal), viewspace_pos, texture_uv
                    )
                    pixel_color = self.fragment_shader(payload)
                    self.set_pixel(x, y, pixel_color)

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
        v1, v2, v3 = vec4(t.a, 1), vec4(t.b, 1), vec4(t.c, 1)
        n1, n2, n3 = vec4(t.n1, 0), vec4(t.n2, 0), vec4(t.n3, 0)
        c1, c2, c3 = t.c1, t.c2, t.c3
        uv1, uv2, uv3 = t.uv1, t.uv2, t.uv3

        n1, n2, n3 = (
            self.model_view_inv[None] @ n1,
            self.model_view_inv[None] @ n2,
            self.model_view_inv[None] @ n3,
        )

        # viewspace positions
        vp1, vp2, vp3 = (
            (self.model_view[None] @ v1).xyz,
            (self.model_view[None] @ v2).xyz,
            (self.model_view[None] @ v3).xyz,
        )

        # model-view-projection
        v1 = self.mvp[None] @ v1
        v2 = self.mvp[None] @ v2
        v3 = self.mvp[None] @ v3

        # Homogeneous division
        v1 /= v1.w
        v2 /= v2.w
        v3 /= v3.w

        # Viewport transformation
        v1 = self.viewport[None] @ v1
        v2 = self.viewport[None] @ v2
        v3 = self.viewport[None] @ v3

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
            normal = vec3(0)
            viewspace_pos = vec3(0)
            texture_uv = vec2(0)
            depth = -tm.inf
            for i, j in ti.ndrange(self.MSAA_N, self.MSAA_N):
                x_curr = x + 2 * D * i + D
                y_curr = y + 2 * D * j + D

                if inside_triangle(x_curr, y_curr, v1.xyz, v2.xyz, v3.xyz):
                    alpha, beta, gamma = compute_barycentric(x_curr, y_curr, v1.xyz, v2.xyz, v3.xyz)
                    w_reciprocal = 1.0 / (alpha / w1 + beta / w2 + gamma / w3)
                    z_interpolated = alpha * z1 / w1 + beta * z2 / w2 + gamma * z3 / w3
                    z_interpolated *= w_reciprocal

                    color += alpha * c1 + beta * c2 + gamma * c3
                    normal += (alpha * n1 + beta * n2 + gamma * n3).xyz
                    viewspace_pos += (alpha * vp1 + beta * vp2 + gamma * vp3).xyz
                    texture_uv += (alpha * uv1 + beta * uv2 + gamma * uv3).xy

                    if depth < z_interpolated:
                        depth = z_interpolated

            if self.depth_buf[x, y] < depth:
                self.depth_buf[x, y] = depth

                color /= self.MSAA_N * self.MSAA_N
                normal /= self.MSAA_N * self.MSAA_N
                viewspace_pos /= self.MSAA_N * self.MSAA_N
                texture_uv /= self.MSAA_N * self.MSAA_N

                # self.set_pixel(x, y, color)
                payload = FragmentShaderPayload(x, y, color, tm.normalize(normal), viewspace_pos, texture_uv)
                pixel_color = self.fragment_shader(payload)
                self.set_pixel(x, y, pixel_color)

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
    texture_png = os.path.abspath("./task/Code/models/spot/spot_texture.png")
    mesh = trimesh.load_mesh(obj_file)
    tex_img = skimage.io.imread(texture_png).astype(np.float32) / 255  # H, W, C

    verts = mesh.vertices.astype(np.float32)
    inds = mesh.faces.astype(np.int32)
    normals = mesh.vertex_normals.astype(np.float32)
    vert_colors = mesh.visual.to_color().vertex_colors.astype(np.float32) / 255
    tex_uvs = mesh.visual.uv.astype(np.float32)

    tex_img = tex_img.transpose(1, 0, 2)[:, ::-1, :]

    # prepare mesh and texture
    mesh = Mesh(verts, inds, normals, vert_colors, tex_uvs)
    texture = Texture(tex_img)

    # create camera
    camera = Camera(pos=(0, 0, 5), up=(0, 1, 0), look_at=(0, 0, -5), fov=60, zNear=0.1, zFar=50)

    # define display
    width = 1024
    height = 1024
    background_color = (0, 0, 0)

    MSAA = False
    MSAA_N = 2  # MSAA-NxN

    # data transform
    data_angles = vec3(0, -140, 0)
    data_scales = vec3(2.5, 2.5, 2.5)
    data_translates = vec3(0, 0, 0)

    # Lighting
    light1 = Light(position=(20, 20, 20), intensity=(500, 500, 500))
    light2 = Light(position=(-20, 20, 0), intensity=(500, 500, 500))

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
    rasterizer.set_texture(texture)
    rasterizer.set_lights(light1, light2)

    # print("viewport:\n", rasterizer.viewport)
    # print("projection:\n", rasterizer.projection)
    # print("view:\n", rasterizer.view)
    # print("model:\n", rasterizer.model)
    # print("mvp:\n", rasterizer.mvp)

    # rendering
    window = ti.ui.Window("draw mesh", rasterizer.resolution)
    canvas = window.get_canvas()

    rasterizer.render()

    angle_delta = 5.0
    curr_angles = copy.deepcopy(data_angles)
    while window.running:
        if window.is_pressed(ti.ui.ESCAPE):  # 按ESC退出
            break

        change_flag = False
        if window.is_pressed(ti.ui.TAB):  # TAB切换 Anti-Aliasing
            rasterizer.MSAA[None] = not rasterizer.MSAA[None]
            change_flag = True

        # 左右键旋转
        if window.is_pressed(ti.ui.LEFT):
            curr_angles[1] = (curr_angles[1] + angle_delta) % 360
            rasterizer.model[None] = get_model_matrix(curr_angles, data_scales, data_translates)
            rasterizer.update_matrices()
            change_flag = True
        if window.is_pressed(ti.ui.RIGHT):
            curr_angles[1] = (curr_angles[1] - angle_delta) % 360
            rasterizer.model[None] = get_model_matrix(curr_angles, data_scales, data_translates)
            rasterizer.update_matrices()
            change_flag = True

        # 选择片段着色器
        if window.is_pressed("q"):
            # 法向贴图（Normal Mapping）
            rasterizer.frag_shader_option[None] = 1
            change_flag = True
        if window.is_pressed("w"):
            # Bling-Phong
            rasterizer.frag_shader_option[None] = 2
            change_flag = True
        if window.is_pressed("e"):
            # 纹理贴图（Texture Mapping）
            rasterizer.frag_shader_option[None] = 3
            change_flag = True
        if window.is_pressed("r"):
            # 位移贴图（Displacement Mapping）
            rasterizer.frag_shader_option[None] = 4
            change_flag = True
        if window.is_pressed("t"):
            # 凹凸贴图（Bump Mapping）
            rasterizer.frag_shader_option[None] = 5
            change_flag = True

        # 空格重置
        if window.is_pressed(ti.ui.SPACE):
            # 使用默认着色器
            rasterizer.frag_shader_option[None] = 0 
            # 重置角度 
            curr_angles = copy.deepcopy(data_angles)  
            rasterizer.model[None] = get_model_matrix(curr_angles, data_scales, data_translates)
            rasterizer.update_matrices()
            change_flag = True

        if change_flag:
            rasterizer.render()

        canvas.set_image(rasterizer.frame_buf)
        window.show()
