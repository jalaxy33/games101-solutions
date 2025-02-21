import os

os.chdir(os.path.dirname(__file__))

import numpy as np
import taichi as ti
import taichi.math as tm
import trimesh
import skimage


ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

# define types
vec2 = ti.types.vector(2, float)
vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)
mat3 = ti.types.matrix(3, 3, float)
mat4 = ti.types.matrix(4, 4, float)


@ti.dataclass
class Camera:
    eye_pos: vec3
    eye_up: vec3
    eye_look_at: vec3
    eye_fov: float
    zNear: float
    zFar: float


@ti.dataclass
class Light:
    position: vec3
    intensity: vec3


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
        self.texture_uvs = ti.Vector.field(3, float, shape=len(texture_uvs))

        self.verticies.from_numpy(vertices.astype(np.float32))
        self.indicies.from_numpy(indicies.astype(np.int32))
        self.normals.from_numpy(normals.astype(np.float32))
        self.vertex_colors.from_numpy(vertex_colors.astype(np.float32))
        self.texture_uvs.from_numpy(texture_uvs.astype(np.float32))


@ti.data_oriented
class Triangle:
    def __init__(self):
        self.verticies = ti.Vector.field(4, float, shape=3)
        self.normals = ti.Vector.field(3, float, shape=3)
        self.colors = ti.Vector.field(3, float, shape=3)
        self.texture_uvs = ti.Vector.field(2, float, shape=3)

        self.a = vec4(0)
        self.b = vec4(0)
        self.c = vec4(0)

    def set_vertices(self, v1: vec4, v2: vec4, v3: vec4):
        self.verticies[0] = v1
        self.verticies[1] = v2
        self.verticies[2] = v3
        self.a = v1
        self.b = v2
        self.c = v3

    def set_normals(self, n1: vec3, n2: vec3, n3: vec3):
        self.normals[0] = n1
        self.normals[1] = n2
        self.normals[2] = n3

    def set_colors(self, c1: vec3, c2: vec3, c3: vec3):
        self.colors[0] = c1
        self.colors[1] = c2
        self.colors[2] = c3

    def set_uvs(self, uv1: vec2, uv2: vec2, uv3: vec2):
        self.texture_uvs[0] = uv1
        self.texture_uvs[1] = uv2
        self.texture_uvs[2] = uv3


@ti.data_oriented
class Rasterizer:
    def __init__(self, width: int, height: int, background_color=vec3(0), MSAA=False, MSAA_N=2):
        self.W = int(width)
        self.H = int(height)
        self.resolution = (self.W, self.H)
        self.aspect_ratio = 1.0 * height / width

        self.background_color = background_color
        self.MSAA = MSAA
        self.MSAA_N = MSAA_N

        self.frame_buf = ti.Vector.field(3, float, shape=self.resolution)
        self.depth_buf = ti.field(float, shape=self.resolution)

    def set_camera(self, camera: Camera):
        self.camera = camera

    def set_lights(self, *lights):
        n = len(lights)
        self.lights = Light.field(shape=n)
        for i in range(n):
            self.lights[i] = lights[i]

    def set_mesh(self, mesh: Mesh):
        self.mesh = mesh

    def set_texture(self, texture: Texture):
        self.texture = texture

    @ti.func
    def set_pixel(self, x: int, y: int, color: ti.template()):
        if x >= 0 and x < self.W and y >= 0 and y < self.H:
            self.frame_buf[x, y] = color

    @ti.kernel
    def render(self):
        self.frame_buf.fill(0)
        self.depth_buf.fill(-tm.inf)

        color = self.texture.get_color(0, 0)
        print(color)


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

    # preprocess
    mesh = Mesh(verts, inds, normals, vert_colors, tex_uvs)
    texture = Texture(tex_img)

    # import matplotlib.pyplot as plt
    # plt.imshow(tex_img)
    # plt.show()

    # print(verts.shape, inds.shape, norms.shape, uvs.shape)
    # mesh.show()

    # camera
    eye_pos = (0, 0, 10)
    eye_up = (0, 1, 0)
    eye_look_at = (0, 0, -5)
    eye_fov = 45
    zNear = 0.1
    zFar = 50

    camera = Camera(
        eye_pos=eye_pos, eye_up=eye_up, eye_look_at=eye_look_at, eye_fov=eye_fov, zNear=zNear, zFar=zFar
    )

    # Lightings
    light1 = Light(position=vec3(20, 20, 20), intensity=(800, 800, 800))
    light2 = Light(position=vec3(-20, 20, 0), intensity=(800, 800, 800))

    kd = vec3(1)  # 漫反射系数
    ks = vec3(0.7937, 0.7937, 0.7937)  # 镜面反射系数
    ka = vec3(0.005, 0.005, 0.005)  # 环境光系数
    ambient_intensity = vec3(10, 10, 10)  # 环境光强度
    phong_exp = 150.0  # 高光区域集中度

    # bump mapping & displacement mapping
    kh = 0.2
    kn = 0.1

    # define display
    width = 1024
    height = 1024
    background_color = vec3(0)

    MSAA = False
    MSAA_N = 2  # MSAA-NxN

    # data transform
    angles = (0, 140, 0)
    scales = vec3(2.5, 2.5, 2.5)
    translates = vec3(0, 0, 0)

    # rendering

    rasterizer = Rasterizer(width, height, background_color, MSAA, MSAA_N)
    rasterizer.set_camera(camera)
    rasterizer.set_lights(light1, light2)
    rasterizer.set_mesh(mesh)
    rasterizer.set_texture(texture)

    rasterizer.render()
