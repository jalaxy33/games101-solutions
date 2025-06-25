import taichi as ti
import numpy as np

from common import *
from Material import *

# from BVH import *
from SahBVH import *


@ti.pyfunc
def ray_triangle_intersect(origin, direction, v1, v2, v3, t_near=Inf):
    # TODO: find ray triangle intersection
    direction = direction.normalized()

    is_hit = False
    hit_t = t_near
    hit_pos = Vec3(Inf)
    hit_norm = Vec3(0)

    # Möller Trumbore Algorithm

    v12 = v2 - v1
    v13 = v3 - v1
    v1o = origin - v1

    S1, S2 = direction.cross(v13), v1o.cross(v12)
    t, u, v = Vec3(S2.dot(v13), S1.dot(v1o), S2.dot(direction)) / S1.dot(v12)

    eps = 1e-9
    if eps < t < t_near and u > eps and v > eps and (1 - u - v) > eps:
        is_hit = True
        hit_t = t
        hit_pos = origin + t * direction
        hit_norm = (v2 - v1).cross(v1 - v3).normalized()

    return is_hit, hit_t, hit_pos, hit_norm


@ti.data_oriented
class Mesh:
    """Basic Mesh"""

    def __init__(self, vertices: NpArr, indices: NpArr):
        self.vertices_np = np.array(vertices, np.float32)
        self.indices_np = np.array(indices, np.int32)

        self.primitive_count = len(self.indices_np)

        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(vertices))
        self.indices = ti.Vector.field(3, dtype=ti.i32, shape=len(indices))

        self.vertices.from_numpy(self.vertices_np)
        self.indices.from_numpy(self.indices_np)

    def set_material(
        self,
        m_type=DIFFUSE_AND_GLOSSY,
        m_color=Vec3(0.5),
        m_emission=Vec3(0),
        ior=1.3,
        kd=Vec3(0.6),
        ks=Vec3(0.1),
        ka=Vec3(0.005),
        spec_exp=0,
    ):
        self.material = Material(m_type, m_color, m_emission, ior, kd, ks, ka, spec_exp)

    @ti.pyfunc
    def hit(self, origin, direction, t_near=Inf):
        """Brute-force search"""

        direction = direction.normalized()

        is_hit = False
        closest_t = t_near
        hit_pos = Vec3(Inf)
        hit_norm = Vec3(0)
        hit_mat = self.material

        for i in range(self.primitive_count):
            i1, i2, i3 = self.indices[i]
            v1, v2, v3 = self.vertices[i1], self.vertices[i2], self.vertices[i3]

            _hit, _t, _pos, _n = ray_triangle_intersect(origin, direction, v1, v2, v3, closest_t)
            if _hit and _t < closest_t:
                is_hit = _hit
                closest_t = _t
                hit_pos = _pos
                hit_norm = _n

        return is_hit, closest_t, hit_pos, hit_norm, hit_mat


@ti.data_oriented
class BvhMesh(Mesh):
    def __init__(self, vertices: NpArr, indices: NpArr):
        super().__init__(vertices, indices)

        # self.bvh = BVH(vertices, indices)
        self.sah = SahBVH(self.vertices_np, self.indices_np)

    # def build_bvh(self, split_method=SplitMethod.MIDDLE, sorter=SorterType.QuickSort):
    #     from time import perf_counter

    #     t0 = perf_counter()
    #     self.bvh.build(split_method, sorter)
    #     t1 = perf_counter()
    #     print(f"Built BVH tree for {self.primitive_count} primitives in {t1 - t0:.3f} s")

    #     self.indices_np = self.indices_np[self.bvh.orders]
    #     self.indices.from_numpy(self.indices_np)

    def build_SAH(self):
        self.sah.build()


    # @ti.pyfunc
    # def hit(self, origin, direction, t_near=Inf):
    #     """BVH search"""

    #     direction = direction.normalized()

    #     is_hit = False
    #     closest_t = t_near
    #     hit_pos = Vec3(Inf)
    #     hit_norm = Vec3(0)
    #     hit_mat = self.material

    #     # curr = self.bvh.nNodes - 1
    #     curr = self.bvh.root
    #     while curr >= 0:
    #         if self.bvh.hit_AABB(curr, origin, direction, closest_t):
    #             if self.bvh.indices[curr] >= 0:
    #                 idx = self.bvh.indices[curr]
    #                 i1, i2, i3 = self.indices[idx]
    #                 v1, v2, v3 = self.vertices[i1], self.vertices[i2], self.vertices[i3]

    #                 _hit, _t, _pos, _n = ray_triangle_intersect(origin, direction, v1, v2, v3, closest_t)
    #                 if _hit:
    #                     is_hit = _hit
    #                     closest_t = _t
    #                     hit_pos = _pos
    #                     hit_norm = _n
    #             curr = self.bvh.hitNexts[curr]
    #         else:
    #             curr = self.bvh.missNexts[curr]

    #     return is_hit, closest_t, hit_pos, hit_norm, hit_mat
