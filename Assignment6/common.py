import taichi as ti
import taichi.math as tm


PI = tm.pi
Inf = float("inf")
iVec2 = ti.types.vector(2, int)
iVec3 = ti.types.vector(3, int)
iVec4 = ti.types.vector(4, int)
Vec2 = ti.types.vector(2, float)
Vec3 = ti.types.vector(3, float)
Vec4 = ti.types.vector(4, float)
Mat2x3 = ti.types.matrix(2, 3, float)
Mat3x2 = ti.types.matrix(3, 2, float)
Mat2x2 = ti.types.matrix(2, 2, float)
Mat3x3 = ti.types.matrix(3, 3, float)
Mat4x4 = ti.types.matrix(4, 4, float)
Mat3x3Id = Mat3x3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
Mat4x4Id = Mat4x4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

# Material Types
DIFFUSE_AND_GLOSSY = 0
REFLECTION_AND_REFRACTION = 1
REFLECTION = 2


@ti.dataclass
class Ray3:
    origin: Vec3
    direct: Vec3

    @ti.func
    def at(self, t: float) -> Vec3:
        return self.origin + self.direct * t


@ti.dataclass
class Material:
    m_type: int  # 材质类型
    m_color: Vec3
    m_emission: Vec3
    ior: float  # 折射系数 refract index
    kd: Vec3  # 漫反射系数
    ks: Vec3  # 镜面反射系数
    ka: Vec3  # 环境光系数
    spec_exp: float  # 高光系数

    def change(
        self,
        m_type=DIFFUSE_AND_GLOSSY,
        m_color=Vec3(0.5),
        m_emission=Vec3(0),
        ior=1.3,
        kd=Vec3(0.6),
        ks=Vec3(0.1),
        ka=Vec3(0.005),
        spec_exp=10.0,
    ):
        self.m_type = m_type
        self.m_color = m_color
        self.m_emission = m_emission
        self.ior = ior
        self.kd = kd
        self.ks = ks
        self.ka = ka
        self.spec_exp = spec_exp


@ti.dataclass
class HitRecord:
    is_hit: bool
    pos: Vec3
    t: float
    N: Vec3  # normal
    front_face: bool
    mat: Material

    @ti.func
    def set_face_normal(self, ray: Ray3, outward_normal: Vec3):
        self.front_face = tm.dot(ray.direct, outward_normal) < 0
        self.N = outward_normal if self.front_face else -outward_normal
