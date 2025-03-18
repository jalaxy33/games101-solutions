import taichi as ti
import taichi.math as tm

# Materials
DIFFUSE_AND_GLOSSY = 0
REFLECTION_AND_REFRACTION = 1
REFLECTION = 2


@ti.dataclass
class Ray:
    ro: tm.vec3  # origin
    rd: tm.vec3  # direction

    @ti.func
    def at(self, t):
        return self.ro + t * self.rd


@ti.dataclass
class Material:
    m_type: int  # 材质类型
    ior: float  # 折射系数 refract index
    kd: tm.vec3  # 漫反射系数
    ks: tm.vec3  # 镜面反射系数
    ka: tm.vec3  # 环境光系数
    diffuse_color: tm.vec3
    spec_exp: float  # 高光系数

    def change(
        self,
        m_type=DIFFUSE_AND_GLOSSY,
        diffuse_color=tm.vec3(0.2),
        ior=1.3,
        kd=tm.vec3(0.8),
        ks=tm.vec3(0.2),
        ka=tm.vec3(0.005),
        spec_exp=25.0,
    ):
        self.m_type = m_type
        self.diffuse_color = diffuse_color
        self.ior = ior
        self.kd = kd
        self.ks = ks
        self.ka = ka
        self.spec_exp = spec_exp


@ti.dataclass
class HitRecord:
    is_hit: bool
    pos: tm.vec3
    t: float
    N: tm.vec3  # normal
    front_face: bool
    mat: Material

    @ti.func
    def set_face_normal(self, ray: Ray, outward_normal: tm.vec3):
        self.front_face = tm.dot(ray.rd, outward_normal) < 0
        self.N = outward_normal if self.front_face else -outward_normal


@ti.data_oriented
class Hittable:
    def __init__(self):
        self.set_material()

    def set_material(
        self,
        m_type=DIFFUSE_AND_GLOSSY,
        diffuse_color=tm.vec3(0.2),
        ior=1.3,
        kd=tm.vec3(0.8),
        ks=tm.vec3(0.2),
        ka=tm.vec3(0.005),
        spec_exp=25.0,
    ):
        self.material = Material(m_type, ior, kd, ks, ka, diffuse_color, spec_exp)

    @ti.func
    def hit(self, ray: Ray, tmin: float) -> HitRecord:
        return HitRecord()


@ti.data_oriented
class Sphere(Hittable):
    def __init__(self, center, radius):
        super().__init__()
        self.c = tm.vec3(center)
        self.r = float(radius)

    @ti.func
    def solve_quadratic(self, a, b, c):
        has_root = False
        x1, x2 = tm.inf, tm.inf

        discr = b * b - 4 * a * c
        if discr >= 0:
            has_root = True
            sqrt_d = tm.sqrt(discr)
            x1 = (-b - sqrt_d) / (2 * a)
            x2 = (-b + sqrt_d) / (2 * a)
        if x1 > x2:
            x1, x2 = x2, x1
        return has_root, x1, x2

    @ti.func
    def hit(self, ray: Ray, tmin: float) -> HitRecord:
        rec = HitRecord()

        oc = ray.ro - self.c
        a = tm.dot(ray.rd, ray.rd)
        b = 2 * tm.dot(ray.rd, oc)
        c = tm.dot(oc, oc) - self.r * self.r

        is_hit, t1, t2 = self.solve_quadratic(a, b, c)
        if t1 < 0:
            t1 = t2
        if t1 < 0:
            is_hit = False

        if is_hit and t1 < tmin:
            rec.is_hit = is_hit
            rec.t = t1
            rec.pos = ray.at(rec.t)
            rec.mat = self.material

            outward_normal = (rec.pos - self.c) / self.r
            rec.set_face_normal(ray, outward_normal)
        return rec


@ti.data_oriented
class MeshTriangle(Hittable):
    def __init__(self, vertices, indices, st_coords):
        self.vertices = ti.Vector.field(3, ti.f32, shape=len(vertices))
        self.indices = ti.Vector.field(3, ti.i32, shape=len(indices))
        self.st_coords = ti.Vector.field(2, ti.f32, shape=len(st_coords))

        self.diffuse_scale = 5
        self.diffuse_color1 = tm.vec3(0.815, 0.235, 0.031)
        self.diffuse_color2 = tm.vec3(0.937, 0.937, 0.231)

        for i in range(len(vertices)):
            self.vertices[i] = vertices[i]
        for i in range(len(indices)):
            self.indices[i] = indices[i]
        for i in range(len(st_coords)):
            self.st_coords[i] = st_coords[i]

    @ti.func
    def lerp(self, a, b, t: float):
        return (1 - t) * a + t * b

    @ti.func
    def eval_diffuse_color(self, st: tm.vec2):
        pattern = (tm.mod(st.x * self.diffuse_scale, 1) > 0.5) ^ (
            tm.mod(st.y * self.diffuse_scale, 1) > 0.5
        )
        return self.lerp(self.diffuse_color1, self.diffuse_color2, pattern)

    @ti.func
    def hit(self, ray: Ray, tmin: float) -> HitRecord:
        rec = HitRecord()

        for index in range(self.indices.shape[0]):
            i1, i2, i3 = self.indices[index]
            v1, v2, v3 = self.vertices[i1], self.vertices[i2], self.vertices[i3]
            st1, st2, st3 = self.st_coords[i1], self.st_coords[i2], self.st_coords[i3]

            # TODO: Implement this function that tests whether the triangle
            # that's specified bt v0, v1 and v2 intersects with the ray (whose
            # origin is *orig* and direction is *dir*)
            # Also don't forget to update tnear, u and v.

            # Möller Trumbore Algorithm:
            e1 = v2 - v1
            e2 = v3 - v1
            s = ray.ro - v1
            s1 = tm.cross(ray.rd, e2)
            s2 = tm.cross(s, e1)

            s1e1 = tm.dot(s1, e1)
            t = tm.dot(s2, e2) / s1e1
            b1 = tm.dot(s1, s) / s1e1  # u
            b2 = tm.dot(s2, ray.rd) / s1e1  # v
            b3 = 1 - b1 - b2

            eps = -1e-9
            if t < tmin and t > eps and b1 > eps and b2 > eps and b3 > eps:
                rec.is_hit = True
                rec.t = t
                rec.pos = ray.at(rec.t)
                rec.mat = self.material

                e1 = e1.normalized()
                e2 = e2.normalized()
                outward_normal = tm.cross(e1, e2)
                rec.set_face_normal(ray, outward_normal)

                st = st1 * b3 + st2 * b1 + st3 * b2
                diffuse_color = self.eval_diffuse_color(st)
                rec.mat.diffuse_color = diffuse_color
        return rec
