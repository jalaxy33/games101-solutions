import taichi as ti
import taichi.math as tm
import numpy as np
from typing import Optional
from time import time

from Material import *
from BVH import BVHTree, SplitMethod


Inf = float("inf")
Vec3 = ti.types.vector(3, dtype=ti.f32)


@ti.data_oriented
class TriMesh:
    def __init__(self, vertices, indices, bvh_split_method=SplitMethod.SAH, brute_force=False):
        self.vertices_np = np.array(vertices, np.float32)
        self.indices_np = np.array(indices, np.int32)

        self.primitive_count = len(self.indices_np)

        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(vertices))
        self.indices = ti.Vector.field(3, dtype=ti.i32, shape=len(indices))

        self.vertices.from_numpy(self.vertices_np)
        self.indices.from_numpy(self.indices_np)

        self.bvh = BVHTree(self.vertices_np[self.indices_np], split_method=bvh_split_method)

        # For Debugging
        self.brute_force = ti.field(ti.i32, shape=())
        self.brute_force[None] = brute_force

    def set_material(
        self,
        m_type=DIFFUSE_AND_GLOSSY,
        m_color=tm.vec3(0.5),
        m_emission=tm.vec3(0),
        ior=1.3,
        kd=tm.vec3(0.6),
        ks=tm.vec3(0.1),
        ka=tm.vec3(0.005),
        spec_exp=0,
    ):
        self.material = Material(m_type, m_color, m_emission, ior, kd, ks, ka, spec_exp)

    def build_bvh(self, split_method: Optional[SplitMethod] = None):
        t0 = time()
        self.bvh.build(split_method)
        t1 = time()
        print(f"Built BVH tree for {self.primitive_count} primitives in {t1 - t0:.3f} s")

        self.indices_np = self.indices_np[self.bvh.order]
        self.indices.from_numpy(self.indices_np)

    ###### Algorithms #######

    @ti.pyfunc
    def hit_triangle(self, index, origin, direction, t_near=Inf):
        
        # TODO: find ray triangle intersection
        direction = direction.normalized()

        is_hit = False
        hit_t = t_near
        hit_pos = ti.Vector([Inf, Inf, Inf], ti.f32)
        hit_norm = ti.Vector([0, 0, 0], ti.f32)

        # Möller Trumbore Algorithm

        idx = self.bvh.index[index]
        if self.brute_force[None]:
            idx = index

        i1, i2, i3 = self.indices[idx]
        a, b, c = self.vertices[i1], self.vertices[i2], self.vertices[i3]

        ab = b - a
        ac = c - a
        ao = origin - a
        n = ab.cross(ac).normalized()

        s1, s2 = direction.cross(ac), ao.cross(ab)
        t, u, v = tm.vec3(s2.dot(ac), s1.dot(ao), s2.dot(direction)) / s1.dot(ab)

        eps = 1e-9
        if eps < t < t_near and u > eps and v > eps and (1 - u - v) > eps:
            is_hit = True
            hit_t = t
            hit_pos = origin + t * direction
            hit_norm = n

        return is_hit, hit_t, hit_pos, hit_norm

    @ti.pyfunc
    def hit(self, origin, direction, t_near=Inf):
        direction = direction.normalized()

        is_hit = False
        hit_t = t_near
        hit_pos = ti.Vector([Inf, Inf, Inf], ti.f32)
        hit_norm = ti.Vector([0, 0, 0], ti.f32)
        hit_mat = self.material

        if self.brute_force[None]:
            is_hit, hit_t, hit_pos, hit_norm, hit_mat = self.brute_force_hit(origin, direction, t_near)
        else:
            is_hit, hit_t, hit_pos, hit_norm, hit_mat = self.bvh_hit(origin, direction, t_near)

        return is_hit, hit_t, hit_pos, hit_norm, hit_mat

    @ti.pyfunc
    def brute_force_hit(self, origin, direction, t_near=Inf):
        direction = direction.normalized()

        is_hit = False
        closest_t = t_near
        hit_pos = ti.Vector([Inf, Inf, Inf], ti.f32)
        hit_norm = ti.Vector([0, 0, 0], ti.f32)
        hit_mat = self.material

        for i in range(self.primitive_count):
            _hit, _t, _pos, _n = self.hit_triangle(i, origin, direction, closest_t)
            if _hit and _t < closest_t:
                is_hit = _hit
                closest_t = _t
                hit_pos = _pos
                hit_norm = _n

        return is_hit, closest_t, hit_pos, hit_norm, hit_mat

    @ti.pyfunc
    def bvh_hit(self, origin, direction, t_near=Inf):
        # TODO: Traverse the BVH to find intersection
        direction = direction.normalized()

        is_hit = False
        closest_t = t_near
        hit_pos = ti.Vector([Inf, Inf, Inf], ti.f32)
        hit_norm = ti.Vector([0, 0, 0], ti.f32)
        hit_mat = self.material

        curr = self.bvh.nNodes - 1
        while curr >= 0:
            if self.bvh.hit_AABB(curr, origin, direction, closest_t):
                if self.bvh.index[curr] >= 0:
                    _hit, _t, _pos, _n = self.hit_triangle(curr, origin, direction, closest_t)
                    if _hit:
                        is_hit = _hit
                        closest_t = _t
                        hit_pos = _pos
                        hit_norm = _n
                curr = self.bvh.hitNext[curr]
            else:
                curr = self.bvh.missNext[curr]
        return is_hit, closest_t, hit_pos, hit_norm, hit_mat
