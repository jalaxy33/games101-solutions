# Homework 3. Shaders & Render Mesh
import os

os.chdir(os.path.dirname(__file__))

import numpy as np
import taichi as ti
import trimesh
import skimage

ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

# local packages
from common import *
from transform import *
from triangle import *


@ti.dataclass
class Camera:
    eye_pos: Vec3
    vup: Vec3
    vfov: float
    look_at: Vec3
    zNear: float
    zFar: float


@ti.dataclass
class Light:
    position: Vec3
    intensity: Vec3


@ti.dataclass
class Material:
    kd: Vec3  # 漫反射系数
    ks: Vec3  # 镜面反射系数
    ka: Vec3  # 环境光系数
    Ia: Vec3  # 环境光强度
    phong_exp: float  # 高光区域集中度
    kh: float  # bump mapping & displacement mapping
    kn: float  # bump mapping & displacement mapping


@ti.data_oriented
class Texture:
    def __init__(self, image: NpArr, name=""):
        self.name = name
        self.W = image.shape[0]
        self.H = image.shape[1]
        self.image = ti.Vector.field(3, float)

        ti.root.dense(ti.ij, (self.W, self.H)).place(self.image)
        self.image.from_numpy(image)

    @ti.pyfunc
    def get_color(self, u: float, v: float):
        i, j = int(u * self.W), int(v * self.H)
        color = Vec3(0)
        if i >= 0 and i < self.W and j >= 0 and j < self.H:
            color = self.image[i, j]
        return color


@ti.data_oriented
class Shader:
    available_shaders = ["default", "normal", "phong", "texture", "displacement", "bump"]

    def __init__(self, shader_option: int = 0):
        self.shader_option = ti.field(int, shape=())
        self.set_shader_option(shader_option)

    def set_shader_option(self, option: int):
        assert isinstance(option, int) and 0 <= option < len(self.available_shaders)
        self.shader_option.fill(option)

    def set_camera(self, camera: Camera):
        self.camera: Camera = camera

    def set_texture(self, texture: Texture):
        self.texture: Texture = texture

    def set_material(self, material: Material):
        self.material: Material = material

    def set_lights(self, lights):
        self.lights = lights

    @ti.func
    def fragment_shader(self, x: int, y: int, color: Vec3, normal: Vec3, viewspace_pos: Vec3, texture_uv: Vec2) -> Vec3:
        pixel_color = color
        if self.shader_option[None] == 1:
            # 法向贴图
            pixel_color = self.normal_fragment_shader(normal)
        elif self.shader_option[None] == 2:
            # Bling-phong
            pixel_color = self.phong_fragment_shader(color, normal, viewspace_pos)
        elif self.shader_option[None] == 3:
            # 纹理映射
            pixel_color = self.texture_fragment_shader(texture_uv, normal, viewspace_pos)
        elif self.shader_option[None] == 4:
            # 位移贴图
            pixel_color = self.displacement_fragment_shader(normal, viewspace_pos, texture_uv)
        elif self.shader_option[None] == 5:
            # 凹凸贴图
            pixel_color = self.bump_fragment_shader(normal, texture_uv)

        return pixel_color

    @ti.func
    def normal_fragment_shader(self, normal: Vec3) -> Vec3:
        """法向贴图"""
        pixel_color = (normal.normalized() + Vec3(1)) / 2
        return pixel_color

    @ti.func
    def phong_fragment_shader(self, color: Vec3, normal: Vec3, viewspace_pos: Vec3) -> Vec3:
        """Bling-phong"""

        pos = viewspace_pos
        normal = normal.normalized()
        eye_pos = self.camera.eye_pos
        kd = color  # 调整漫反射系数为颜色

        pixel_color = Vec3(0)
        for i in ti.ndrange(self.lights.shape[0]):
            # TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular*
            # components are. Then, accumulate that result on the *result_color* object.

            light = self.lights[i]
            light_pos = light.position.xyz
            light_intensity = light.intensity.xyz

            r = (light_pos - pos).norm()  # 光源-点距离
            light_dir = (light_pos - pos).normalized()  # 光源-点方向
            view_dir = (eye_pos - pos).normalized()  # 相机-点方向
            half_dir = (view_dir + light_dir).normalized()  # 半程向量

            E = light_intensity / (r * r)  # 此处的光能

            ambient = self.material.ka * self.material.Ia
            diffuse = kd * E * ti.max(0, normal.dot(light_dir))
            specular = self.material.ks * E * ti.pow(ti.max(0, normal.dot(half_dir)), self.material.phong_exp)

            pixel_color += ambient + diffuse + specular
        return pixel_color

    @ti.func
    def texture_fragment_shader(self, texture_uv: Vec2, normal: Vec3, viewspace_pos: Vec3) -> Vec3:
        """纹理映射"""

        # TODO: Get the texture value at the texture coordinates of the current fragment
        u, v = texture_uv
        color = self.texture.get_color(u, v)
        pixel_color = self.phong_fragment_shader(color, normal, viewspace_pos)
        return pixel_color

    @ti.func
    def displacement_fragment_shader(self, normal: Vec3, viewspace_pos: Vec3, texture_uv: Vec2) -> Vec3:
        """位移贴图"""

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

        n = normal.normalized()
        nx, ny, nz = n.xyz
        u, v = texture_uv.xy
        W, H = self.texture.W, self.texture.H

        sqrt_xz = ti.sqrt(nx * nx + nz * nz)
        t = Vec3(nx * ny / sqrt_xz, sqrt_xz, nz * ny / sqrt_xz)
        b = n.cross(t)

        TBN = Mat3x3(t, b, n).transpose()
        color = self.texture.get_color(u, v)

        dU = self.material.kh * self.material.kn
        dU *= self.texture.get_color(u + 1.0 / W, v).norm() - color.norm()
        dV = self.material.kh * self.material.kn
        dV *= self.texture.get_color(u, v + 1.0 / H).norm() - color.norm()
        ln = Vec3(-dU, -dV, 1)

        vp = viewspace_pos + self.material.kn * n * color.norm()
        n_ = (TBN @ ln).normalized()
        pixel_color = self.phong_fragment_shader(color, n_, vp)
        return pixel_color

    @ti.func
    def bump_fragment_shader(self, normal: Vec3, texture_uv: Vec2) -> Vec3:
        """凹凸贴图"""

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

        n = normal.normalized()
        nx, ny, nz = n.xyz
        u, v = texture_uv.xy
        W, H = self.texture.W, self.texture.H

        n = normal.normalized()
        nx, ny, nz = n.xyz
        u, v = texture_uv.xy
        W, H = self.texture.W, self.texture.H

        sqrt_xz = ti.sqrt(nx * nx + nz * nz)
        t = Vec3(nx * ny / sqrt_xz, sqrt_xz, nz * ny / sqrt_xz)
        b = n.cross(t)

        TBN = Mat3x3(t, b, n).transpose()
        color = self.texture.get_color(u, v)

        dU = self.material.kh * self.material.kn
        dU *= self.texture.get_color(u + 1.0 / W, v).norm() - color.norm()
        dV = self.material.kh * self.material.kn
        dV *= self.texture.get_color(u, v + 1.0 / H).norm() - color.norm()
        ln = Vec3(-dU, -dV, 1)

        pixel_color = (TBN @ ln).normalized()
        return pixel_color


@ti.data_oriented
class Renderer:
    def __init__(
        self,
        width: int,
        height: int,
        background_color=(0, 0, 0),
        MSAA: bool = False,
        MSAA_N: int = 2,
        shader_option: int = 0,
    ):
        self.W = int(width)
        self.H = int(height)
        self.resolution = (self.W, self.H)
        self.aspect_ratio = 1.0 * height / width

        self.background_color = Vec3(background_color)

        self.MSAA = ti.field(int, shape=())
        self.MSAA_N = ti.field(int, shape=())
        self.MSAA.fill(MSAA)
        self.MSAA_N.fill(MSAA_N)

        self.shader = Shader(shader_option)

        self.frame_buf = ti.Vector.field(3, float, shape=self.resolution)  # 屏幕像素颜色信息（rbg）
        self.depth_buf = ti.field(float, shape=self.resolution)  # 屏幕像素深度信息（z-buffer）

        # transform matrices
        self.viewport = self.init_Mat4x4_field()
        self.view = self.init_Mat4x4_field()
        self.projection = self.init_Mat4x4_field()
        self.model = self.init_Mat4x4_field()

        self.mvp = self.init_Mat4x4_field()
        self.model_view = self.init_Mat4x4_field()
        self.model_view_inv = self.init_Mat4x4_field()

    def init_Mat4x4_field(self):
        matrix = Mat4x4.field(shape=())
        matrix.fill(eye4())
        return matrix

    def set_camera(self, eye_pos: Vec3, vup: Vec3, vfov: float, look_at: Vec3, zNear: float, zFar: float):
        self.camera = Camera(eye_pos, vup, vfov, look_at, zNear, zFar)

        self.shader.set_camera(self.camera)

        self.viewport.fill(get_viewport_matrix(self.W, self.H))
        self.view.fill(get_view_matrix(self.camera.eye_pos, self.camera.look_at, self.camera.vup))
        self.projection.fill(
            get_projection_matrix(self.camera.vfov, self.aspect_ratio, self.camera.zNear, self.camera.zFar)
        )
        self.model.fill(get_model_matrix())
        self.mvp.fill(self.viewport[None] @ self.projection[None] @ self.view[None] @ self.model[None])
        self.model_view.fill(self.view[None] @ self.model[None])
        self.model_view_inv.fill(self.model_view[None].inverse().transpose())

        # print("viewport:\n", self.viewport)
        # print("view:\n", self.view)
        # print("projection:\n", self.projection)
        # print("model:\n", self.model)
        # print("mvp:\n", self.mvp)
        # print("model_view:\n", self.model_view)
        # print("model_view_inv:\n", self.model_view_inv)

    def set_model_transform(self, angles=(0, 0, 0), scales=(1, 1, 1), translates=(0, 0, 0)):
        self.model.fill(get_model_matrix(angles, scales, translates))
        self.mvp.fill(self.viewport[None] @ self.projection[None] @ self.view[None] @ self.model[None])
        self.model_view.fill(self.view[None] @ self.model[None])
        self.model_view_inv.fill(self.model_view[None].inverse().transpose())

        # print("After model transform:\n")
        # print("model:\n", self.model)
        # print("mvp:\n", self.mvp)
        # print("model_view:\n", self.model_view)
        # print("model_view_inv:\n", self.model_view_inv)

    def set_mesh(self, vertices: NpArr, indices: NpArr, normals: NpArr, vertex_colors: NpArr, texture_uvs: NpArr):
        self.vertices = ti.Vector.field(3, float)
        self.indices = ti.Vector.field(3, int)
        self.normals = ti.Vector.field(3, float)
        self.vertex_colors = ti.Vector.field(3, float)
        self.texture_uvs = ti.Vector.field(2, float)

        ti.root.dense(ti.i, len(vertices)).place(self.vertices)
        ti.root.dense(ti.i, len(indices)).place(self.indices)
        ti.root.dense(ti.i, len(normals)).place(self.normals)
        ti.root.dense(ti.i, len(vertex_colors)).place(self.vertex_colors)
        ti.root.dense(ti.i, len(texture_uvs)).place(self.texture_uvs)

        self.vertices.from_numpy(vertices)
        self.indices.from_numpy(indices)
        self.normals.from_numpy(normals)
        self.vertex_colors.from_numpy(vertex_colors)
        self.texture_uvs.from_numpy(texture_uvs)

    def set_texture(self, image: NpArr):
        self.texture = Texture(image)
        self.shader.set_texture(self.texture)

    def set_material(self, kd: Vec3, ks: Vec3, ka: Vec3, Ia: Vec3, phog_exp: float, kh: float, kn: float):
        self.material = Material(kd, ks, ka, Ia, phog_exp, kh, kn)
        self.shader.set_material(self.material)

    def set_lights(self, *lights):
        assert all([isinstance(l, Light) for l in lights])

        self.lights = Light.field()
        ti.root.dense(ti.i, len(lights)).place(self.lights)
        for i in range(len(lights)):
            self.lights[i] = lights[i]

        self.shader.set_lights(self.lights)

    def set_shader_option(self, option: int):
        self.shader.set_shader_option(option)

    @ti.kernel
    def render(self):
        self.frame_buf.fill(self.background_color)
        self.depth_buf.fill(-INF)

        for i in self.indices:
            i1, i2, i3 = self.indices[i]

            if self.MSAA[None]:
                self.rasterize_triangle_MSAA(i1, i2, i3)
            else:
                self.rasterize_triangle(i1, i2, i3)

    @ti.func
    def set_pixel(self, x: int, y: int, color: ti.template()):
        if 0 <= x < self.W and 0 <= y < self.H:
            self.frame_buf[x, y] = color

    @ti.func
    def rasterize_triangle(self, i1: int, i2: int, i3: int):
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

        v1, v2, v3 = Vec4(self.vertices[i1], 1), Vec4(self.vertices[i2], 1), Vec4(self.vertices[i3], 1)
        n1, n2, n3 = (
            self.model_view_inv[None] @ Vec4(self.normals[i1].normalized(), 0),
            self.model_view_inv[None] @ Vec4(self.normals[i2].normalized(), 0),
            self.model_view_inv[None] @ Vec4(self.normals[i3].normalized(), 0),
        )
        c1, c2, c3 = self.vertex_colors[i1], self.vertex_colors[i2], self.vertex_colors[i3]
        uv1, uv2, uv3 = self.texture_uvs[i1], self.texture_uvs[i2], self.texture_uvs[i3]
        # viewspace positions
        vp1, vp2, vp3 = (
            self.model_view[None] @ v1,
            self.model_view[None] @ v2,
            self.model_view[None] @ v3,
        )

        v1 = self.mvp[None] @ v1
        v2 = self.mvp[None] @ v2
        v3 = self.mvp[None] @ v3

        v1 /= v1.w
        v2 /= v2.w
        v3 /= v3.w

        lbound, ubound = get_triangle_bbox(v1, v2, v3)
        x_min, x_max = int(ti.floor(lbound.x)), int(ti.ceil(ubound.x))
        y_min, y_max = int(ti.floor(lbound.y)), int(ti.ceil(ubound.y))

        z1, w1 = v1.zw
        z2, w2 = v2.zw
        z3, w3 = v3.zw

        for x, y in ti.ndrange((x_min, x_max + 1), (y_min, y_max + 1)):
            if inside_triangle(x, y, v1.xy, v2.xy, v3.xy):
                alpha, beta, gamma = compute_barycentric(x, y, v1.xy, v2.xy, v3.xy)
                w_reciprocal = 1.0 / (alpha / w1 + beta / w2 + gamma / w3)
                z_interpolated = alpha * z1 / w1 + beta * z2 / w2 + gamma * z3 / w3
                z_interpolated *= w_reciprocal

                if self.depth_buf[x, y] < z_interpolated:
                    self.depth_buf[x, y] = z_interpolated

                    color = (alpha * c1 + beta * c2 + gamma * c3).xyz
                    normal = (alpha * n1 + beta * n2 + gamma * n3).xyz.normalized()
                    viewspace_pos = (alpha * vp1 + beta * vp2 + gamma * vp3).xyz
                    texture_uv = (alpha * uv1 + beta * uv2 + gamma * uv3).xy

                    pixel_color = self.shader.fragment_shader(x, y, color, normal, viewspace_pos, texture_uv)
                    self.set_pixel(x, y, pixel_color)

    @ti.func
    def rasterize_triangle_MSAA(self, i1: int, i2: int, i3: int):
        # TODO: Implement MSAA version

        v1, v2, v3 = Vec4(self.vertices[i1], 1), Vec4(self.vertices[i2], 1), Vec4(self.vertices[i3], 1)
        n1, n2, n3 = (
            self.model_view_inv[None] @ Vec4(self.normals[i1].normalized(), 0),
            self.model_view_inv[None] @ Vec4(self.normals[i2].normalized(), 0),
            self.model_view_inv[None] @ Vec4(self.normals[i3].normalized(), 0),
        )
        c1, c2, c3 = self.vertex_colors[i1], self.vertex_colors[i2], self.vertex_colors[i3]
        uv1, uv2, uv3 = self.texture_uvs[i1], self.texture_uvs[i2], self.texture_uvs[i3]
        # viewspace positions
        vp1, vp2, vp3 = (
            self.model_view[None] @ v1,
            self.model_view[None] @ v2,
            self.model_view[None] @ v3,
        )

        v1 = self.mvp[None] @ v1
        v2 = self.mvp[None] @ v2
        v3 = self.mvp[None] @ v3

        v1 /= v1.w
        v2 /= v2.w
        v3 /= v3.w

        lbound, ubound = get_triangle_bbox(v1, v2, v3)
        x_min, x_max = int(ti.floor(lbound.x)), int(ti.ceil(ubound.x))
        y_min, y_max = int(ti.floor(lbound.y)), int(ti.ceil(ubound.y))

        z1, w1 = v1.zw
        z2, w2 = v2.zw
        z3, w3 = v3.zw

        D = 1.0 / (2 * self.MSAA_N[None])
        scale_ratio = 1.0 / (self.MSAA_N[None] * self.MSAA_N[None])
        for x, y in ti.ndrange((x_min, x_max + 1), (y_min, y_max + 1)):
            color = Vec3(0)
            normal = Vec3(0)
            viewspace_pos = Vec3(0)
            texture_uv = Vec2(0)
            depth = -INF
            for i, j in ti.ndrange(self.MSAA_N[None], self.MSAA_N[None]):
                x_curr = x + 2 * D * i + D
                y_curr = y + 2 * D * j + D

                if inside_triangle(x, y, v1.xy, v2.xy, v3.xy):
                    alpha, beta, gamma = compute_barycentric(x_curr, y_curr, v1.xy, v2.xy, v3.xy)
                    w_reciprocal = 1.0 / (alpha / w1 + beta / w2 + gamma / w3)
                    z_interpolated = alpha * z1 / w1 + beta * z2 / w2 + gamma * z3 / w3
                    z_interpolated *= w_reciprocal

                    color += (alpha * c1 + beta * c2 + gamma * c3).xyz
                    normal += (alpha * n1 + beta * n2 + gamma * n3).xyz
                    viewspace_pos += (alpha * vp1 + beta * vp2 + gamma * vp3).xyz
                    texture_uv += (alpha * uv1 + beta * uv2 + gamma * uv3).xy

                    if depth < z_interpolated:
                        depth = z_interpolated

            if self.depth_buf[x, y] < depth:
                self.depth_buf[x, y] = depth

                color = color * scale_ratio
                normal = (normal * scale_ratio).normalized()
                viewspace_pos = viewspace_pos * scale_ratio
                texture_uv = texture_uv * scale_ratio

                pixel_color = self.shader.fragment_shader(x, y, color, normal, viewspace_pos, texture_uv)
                self.set_pixel(x, y, pixel_color)


if __name__ == "__main__":
    # load model
    obj_file = os.path.abspath("../task/Code/models/spot/spot_triangulated_good.obj")
    texture_png = os.path.abspath("../task/Code/models/spot/spot_texture.png")
    mesh = trimesh.load_mesh(obj_file)
    texture_img = skimage.io.imread(texture_png).astype(np.float32) / 255  # H, W, C

    vertices = mesh.vertices.astype(np.float32)
    indices = mesh.faces.astype(np.int32)
    normals = mesh.vertex_normals.astype(np.float32)
    vertex_colors = mesh.visual.to_color().vertex_colors.astype(np.float32) / 255
    texture_uvs = mesh.visual.uv.astype(np.float32)

    texture_img = texture_img.transpose(1, 0, 2)[:, ::-1, :]

    # set lights
    light1 = Light(position=(20, 20, 20), intensity=(500, 500, 500))
    light2 = Light(position=(-20, 20, 0), intensity=(500, 500, 500))

    # create renderer
    renderer = Renderer(
        width=1024,
        height=1024,
        background_color=(0, 0, 0),
        MSAA=False,
        MSAA_N=2,
        shader_option=3,
    )
    renderer.set_camera(eye_pos=(0, 0, 5), vup=(0, 1, 0), vfov=60, look_at=(0, 0, -5), zNear=0.1, zFar=50)
    renderer.set_material(
        kd=(1, 1, 1),  # 漫反射系数
        ks=(0.7937, 0.7937, 0.7937),  # 镜面反射系数
        ka=(0.005, 0.005, 0.005),  # 环境光系数
        Ia=(10, 10, 10),  # 环境光强度
        phog_exp=150.0,  # 高光区域集中度
        kh=0.2 * 15,  # bump mapping & displacement mapping
        kn=0.1 * 15,  # bump mapping & displacement mapping
    )
    renderer.set_mesh(vertices, indices, normals, vertex_colors, texture_uvs)
    renderer.set_texture(texture_img)
    renderer.set_lights(light1, light2)

    # set initial model transform
    renderer.set_model_transform(angles=(0, -140, 0), scales=(2.5, 2.5, 2.5), translates=(0, 0, 0))

    # rendering
    window = ti.ui.Window("draw mesh", renderer.resolution)
    canvas = window.get_canvas()
    gui = window.get_gui()

    renderer.render()

    angle_delta = 5.0
    hold_tab_count = 0
    hold_space_count = 0
    max_hold = 12
    while window.running and not window.is_pressed(ti.ui.ESCAPE):
        shader_option = renderer.shader.shader_option[None]
        current_shader = renderer.shader.available_shaders[shader_option]
        MSAA_status = "True" if renderer.MSAA[None] else "False"
        MSAA_N = renderer.MSAA_N[None]

        gui.text(f"Shader: {current_shader}")
        gui.text(f"MSAA: {MSAA_status}")
        gui.text(f"MSAA_N: {MSAA_N}")

        switch_flag = False
        if window.is_pressed(ti.ui.TAB):  # 按 TAB 切换 Shader
            hold_tab_count += 1
            if hold_tab_count >= max_hold:
                hold_tab_count = 0
                switch_flag = True
                shader_option = (shader_option + 1) % len(renderer.shader.available_shaders)
                renderer.set_shader_option(shader_option)

        if window.is_pressed(ti.ui.CAPSLOCK):  # 按 CAPSLOCK 切换 MSAA
            hold_space_count += 1
            if hold_space_count >= max_hold:
                hold_space_count = 0
                switch_flag = True
                renderer.MSAA[None] = not renderer.MSAA[None]

        if switch_flag:
            renderer.render()

        canvas.set_image(renderer.frame_buf)
        window.show()
