import taichi as ti

from common import Vec3, Vec4


@ti.pyfunc
def barycentric_ij(p: ti.template(), vi: ti.template(), vj: ti.template()) -> float:
    return (vi.y - vj.y) * p.x + (vj.x - vi.x) * p.y + (vi.x * vj.y - vj.x * vi.y)


@ti.dataclass
class Triangle:
    # vertex
    v1: Vec4
    v2: Vec4
    v3: Vec4
    # vertex color
    c1: Vec3
    c2: Vec3
    c3: Vec3

    @ti.pyfunc
    def get_bounds(self) -> tuple[Vec4, Vec4]:
        lb = ti.min(self.v1, self.v2, self.v3)
        ub = ti.max(self.v1, self.v2, self.v3)
        return lb, ub

    @ti.pyfunc
    def inside(self, p: ti.template()) -> bool:
        p_ = Vec3(p.x, p.y, 1)
        a = self.v1.xyz
        b = self.v2.xyz
        c = self.v3.xyz

        f1 = (p_ - a).cross(b - a).z > 0
        f2 = (p_ - b).cross(c - b).z > 0
        f3 = (p_ - c).cross(a - c).z > 0
        return (f1 == f2) and (f2 == f3)

    @ti.pyfunc
    def compute_barycentric(self, p: ti.template()) -> Vec3:
        alpha = barycentric_ij(p, self.v2, self.v3)
        beta = barycentric_ij(p, self.v3, self.v1)
        # gamma = barycentric_ij(p, self.v1, self.v2)
        gamma = 1 - alpha - beta
        return Vec3(alpha, beta, gamma)


    @ti.pyfunc
    def interpolate_z_and_color(self, p:ti.template()):
        z1, z2, z3 = self.v1.z, self.v2.z, self.v3.z
        w1, w2, w3 = self.v1.w, self.v2.w, self.v3.w
        c1, c2, c3 = self.c1, self.c2, self.c3

        alpha, beta, gamma = self.compute_barycentric(p)

        w_reciprocal = 1.0 / (alpha / w1 + beta / w2 + gamma / w3)
        z_interp = alpha * z1 / w1 + beta * z2 / w2 + gamma * z3 / w3
        z_interp *= w_reciprocal

        color_interp = alpha * c1 + beta * c2 + gamma * c3
        return z_interp, color_interp