# Assignment 6. BVH RayTacer
# Reference: https://github.com/Jiay1C/path_tracing_obj/blob/master/path_tracing_obj.py

import os

os.chdir(os.path.dirname(__file__))


from vispy import io as vispyio
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)


vec3 = ti.types.vector(3, ti.f32)
vec4 = ti.types.vector(4, ti.f32)


# Args
WIDTH = 1280
HEIGHT = 960

# Materials
DIFFUSE_AND_GLOSSY = 1
REFLECTION_AND_REFRACTION = 2
REFLECTION = 3


# Global Variables
FrameBuffer = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))


@ti.dataclass
class Material:
    type: int
    color: vec3
    emission: vec3
    kd: float
    ks: float
    specular_p: float
    refract_idx: float


@ti.data_oriented
class Model:
    def __init__(self, max_vertex_num=1024, max_face_num=1024):
        self.vertex_num = ti.field(dtype=ti.i32, shape=())
        self.vertex_num[None] = 0
        self.face_num = ti.field(dtype=ti.i32, shape=())
        self.face_num[None] = 0

        self.vertices = ti.Vector.field(4, dtype=ti.f32, shape=max_vertex_num)
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=max_face_num)
        self.normals = ti.Vector.field(3, dtype=ti.f32, shape=max_vertex_num)
        self.texcoords = ti.Vector.field(2, dtype=ti.f32, shape=max_vertex_num)

        self.set_material(
            type=DIFFUSE_AND_GLOSSY,
            color=(1, 1, 1),
            emission=(0, 0, 0),
            kd=0.6,
            ks=0.0,
            specular_p=0.0,
            refract_idx=1.0,
        )

    def set_material(
        self,
        type=DIFFUSE_AND_GLOSSY,
        color=(1, 1, 1),
        emission=(0, 0, 0),
        kd=0.6,
        ks=0.0,
        specular_p=0.0,
        refract_idx=1.0,
    ):
        self.material = Material(type, color, emission, kd, ks, specular_p, refract_idx)

    def clear_face(self):
        self.vertex_num[None] = 0
        self.face_num[None] = 0

    def from_obj(self, filename):
        self.clear_face()
        vertices, faces, normals, texcoords = vispyio.read_mesh(filename)
        for index in range(len(vertices)):
            self.vertices[index] = ti.Vector([vertices[index][0], vertices[index][1], vertices[index][2], 1])
        for index in range(len(faces)):
            self.faces[index] = ti.Vector([faces[index][0], faces[index][1], faces[index][2]])
        for index in range(len(normals)):
            self.normals[index] = ti.Vector([normals[index][0], normals[index][1], normals[index][2]])

        if texcoords:
            for index in range(len(texcoords)):
                self.texcoords[index] = ti.Vector([texcoords[index][0], texcoords[index][1]])

        self.vertex_num[None] = len(vertices)
        self.face_num[None] = len(faces)

    def transform(self, matrix):
        for index in range(self.vertex_num[None]):
            self.vertices[index] = matrix @ self.vertices[index]


@ti.data_oriented
class Scene:
    def __init__(self):
        self.models = []

    def add_model(self, model):
        self.models.append(model)

    def clear_model(self):
        self.models.clear()


scene = Scene()


@ti.func
def reflect(v, normal):
    return v - 2 * v.dot(normal) * normal


@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = min(n.dot(-uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel


@ti.func
def reflectance(cosine, ref_idx):
    # Use Schlick's approximation for reflectance.
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)


def scene_init_bunny():
    bunny = Model(max_vertex_num=2503, max_face_num=4968)
    bunny.from_obj("./task/Assignment6/models/bunny/bunny.obj")
    scene.add_model(bunny)


if __name__ == "__main__":
    window = ti.ui.Window("BVH RayTacer", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    FrameBuffer.fill(0)
    scene_init_bunny()
    while window.running:
        if window.is_pressed(ti.ui.ESCAPE):  # Press <ESC> to exit
            break

        canvas.set_image(FrameBuffer)
        window.show()
