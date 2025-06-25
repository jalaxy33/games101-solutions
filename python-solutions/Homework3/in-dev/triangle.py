import taichi as ti

from common import *


@ti.pyfunc
def get_triangle_bbox(v1, v2, v3):
    lbound = ti.min(v1, v2, v3)
    ubound = ti.max(v1, v2, v3)
    return lbound, ubound


@ti.pyfunc
def inside_triangle(x: float, y: float, a: Vec2, b: Vec2, c: Vec2) -> bool:
    p = Vec3(x, y, 0)
    v1 = Vec3(a.xy, 0)
    v2 = Vec3(b.xy, 0)
    v3 = Vec3(c.xy, 0)

    f1 = (p - v1).cross(v2 - v1).z > 0
    f2 = (p - v2).cross(v3 - v2).z > 0
    f3 = (p - v3).cross(v1 - v3).z > 0
    return (f1 == f2) and (f2 == f3)


@ti.pyfunc
def barycentric_ij(x: float, y: float, a: Vec2, b: Vec2) -> float:
    return (a.y - b.y) * x + (b.x - a.x) * y + a.x * b.y - b.x * a.y


# 计算三角形重心坐标
@ti.pyfunc
def compute_barycentric(x: float, y: float, a: Vec2, b: Vec2, c: Vec2) -> Vec3:
    alpha = barycentric_ij(x, y, b, c) / barycentric_ij(a.x, a.y, b, c)
    beta = barycentric_ij(x, y, c, a) / barycentric_ij(b.x, b.y, c, a)
    # gamma = barycentric_ij(x, y, a, b) / barycentric_ij(c.x, c.y, a, b)
    gamma = 1 - alpha - beta
    return Vec3(alpha, beta, gamma)
