import numpy as np
import taichi as ti
import taichi.math as tm


# Material Types
DIFFUSE_AND_GLOSSY = 0
REFLECTION_AND_REFRACTION = 1
REFLECTION = 2


@ti.dataclass
class Material:
    m_type: int
    kd: tm.vec3  # 漫反射系数
    ks: tm.vec3  # 镜面反射系数
    spec_exp: float  # 高光指数
    refrac_idx: float  # 折射系数
    diffuse_color: tm.vec3  # 材质颜色


@ti.data_oriented
class Model:
    def __init__(self):
        self.material = Material(
            m_type=DIFFUSE_AND_GLOSSY,
            kd=tm.vec3(0.8),
            ks=tm.vec3(0.2),
            spec_exp=25.0,
            refrac_idx=1.3,
            diffuse_color=(0.2, 0.2, 0.2),
        )

    def set_material(
        self,
        m_type=DIFFUSE_AND_GLOSSY,
        kd=tm.vec3(0.8),
        ks=tm.vec3(0.2),
        spec_exp=25.0,
        refrac_idx=1.3,
        diffuse_color=(0.2, 0.2, 0.2),
    ):
        self.material = Material(m_type, kd, ks, spec_exp, refrac_idx, diffuse_color)


@ti.func
def solve_quadratic(a, b, c):
    has_root = False
    x1, x2 = tm.inf, tm.inf

    discr = b * b - 4 * a * c
    sqrt_d = tm.sqrt(discr)
    if discr >= 0:
        has_root = True
        x1 = (-b - sqrt_d) / (2 * a)
        x2 = (-b + sqrt_d) / (2 * a)
    if x1 > x2:
        x1, x2 = x2, x1
    return has_root, x1, x2


@ti.data_oriented
class Sphere(Model):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    @ti.func
    def hit(self, ray_orig, ray_dir, t_min: float):
        C, R = self.center, self.radius
        O, D = ray_orig, ray_dir

        L = O - C
        a = D.dot(D)
        b = 2 * D.dot(L)
        c = L.dot(L) - R * R
        is_hit, t1, t2 = solve_quadratic(a, b, c)
        if t1 < 0:
            t1 = t2
        if t1 < 0:
            is_hit = False
        t_closest = t1

        N = (O - C).normalize()
        diffuse_color = self.material.diffuse_color
        return is_hit, t_closest, N, diffuse_color


@ti.func
def lerp(v1, v2, ratio: float):
    # frac: [0.0, 1.0]
    assert ratio >= 0.0 and ratio <= 1.0
    return v1 + ratio * (v2 - v1)


@ti.data_oriented
class MeshTriangle(Model):
    def __init__(self, vertices, indices, st_coords):
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(vertices))
        self.indices = ti.Vector.field(3, dtype=ti.i32, shape=len(indices))
        self.st_coords = ti.Vector.field(2, dtype=ti.f32, shape=len(st_coords))

        self.vertices.from_numpy(np.array(vertices, np.float32))
        self.indices.from_numpy(np.array(indices, np.int32))
        self.st_coords.from_numpy(np.array(st_coords, np.float32))

        self.color_scale = 5
        self.color1 = tm.vec3(0.815, 0.235, 0.031)
        self.color2 = tm.vec3(0.937, 0.937, 0.231)

    @ti.func
    def hit(self, ray_orig, ray_dir, t_min: float):
        O, D = ray_orig, ray_dir

        is_hit = False
        t_closest = t_min
        N = tm.vec3(0)
        diffuse_color = self.material.diffuse_color

        # Moller Trumbore Algorithm
        for index in range(self.indices.shape[0]):
            # TODO: Implement this function that tests whether the triangle
            # that's specified bt v0, v1 and v2 intersects with the ray (whose
            # origin is *orig* and direction is *dir*)
            # Also don't forget to update tnear, u and v.
            i1, i2, i3 = self.indices[index]
            v1, v2, v3 = self.vertices[i1], self.vertices[i2], self.vertices[i3]
            st1, st2, st3 = self.st_coords[i1], self.st_coords[i2], self.st_coords[i3]

            e1 = v2 - v1
            e2 = v3 - v1
            s = O - v1

            s1 = D.cross(e2)
            s2 = s.cross(e1)

            s1e1 = s1.dot(e1)
            t = s2.dot(e2) / s1e1
            b1 = s1.dot(s) / s1e1  # u
            b2 = s2.dot(D) / s1e1  # v

            eps = 1e-9
            if t < t_closest and b1 > eps and b2 > eps and (1 - b1 - b2) > eps:
                is_hit = True
                t_closest = t
                st = st1 * (1 - b1 - b2) + st2 * b1 + st3 * b2
                diffuse_color = self.eval_diffuse_color(st.x, st.y)

        return is_hit, t_closest, N, diffuse_color

    def eval_diffuse_color(self, u: float, v: float):
        pattern = (tm.mod(u * self.color_scale, 1) > 0.5) ^ (tm.mod(v * self.color_scale, 1) > 0.5)
        return lerp(self.color1, self.color2, pattern)
