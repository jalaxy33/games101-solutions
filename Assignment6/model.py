# reference: https://gitee.com/mmoon/taichi-rt/blob/master/ti_rt/utils/model.py

import taichi as ti
import taichi.math as tm
import numpy as np
from time import time

from common import *
from bvh import *


class Hittable:
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


@ti.data_oriented
class BVHMesh(Hittable):
    def __init__(self, vertices, indices, normals, split_method=SplitMethod.SAH, mix_norm=False):
        self.verts_cpu = np.array(vertices, np.float32)
        self.inds_cpu = np.array(indices, np.int32)
        self.norms_cpu = np.array(normals, np.float32)
        self.mix_norm = mix_norm

        self.bvh = BVHTree(self.verts_cpu[self.inds_cpu], split_method)

    def build_bvh(self, split_method: Optional[SplitMethod] = None):
        t0 = time()
        self.bvh.build(split_method)
        t1 = time()
        print(f"Built BVH tree for {len(self.inds_cpu)} primitives in {t1 - t0:.3f} s")

        self.inds_cpu = self.inds_cpu[self.bvh.order]

        self.vertices = ti.Vector.field(3, ti.f32, shape=len(self.verts_cpu))
        self.indices = ti.Vector.field(3, ti.i32, shape=len(self.inds_cpu))
        self.normals = ti.Vector.field(3, ti.f32, shape=len(self.norms_cpu))

        self.vertices.from_numpy(self.verts_cpu)
        self.indices.from_numpy(self.inds_cpu)
        self.normals.from_numpy(self.norms_cpu)

    @ti.func
    def hit_triangle(self, curr: int, ray: Ray3, tmin: float) -> HitRecord:
        # TODO: find ray triangle intersection

        # Möller Trumbore Algorithm
        i1, i2, i3 = self.indices[self.bvh.index[curr]]
        v1, v2, v3 = self.vertices[i1], self.vertices[i2], self.vertices[i3]
        n1, n2, n3 = self.normals[i1], self.normals[i2], self.normals[i3]

        T, E1, E2 = ray.origin - v1, v2 - v1, v3 - v1
        P, Q = ray.direct.cross(E2), T.cross(E1)
        t, u, v = Vec3(Q.dot(E2), P.dot(T), Q.dot(ray.direct)) / P.dot(E1)

        eps = 1e-9
        is_hit = eps < t < tmin and u > eps and v > eps and 1 - u - v > eps

        rec = HitRecord()
        if is_hit:
            rec.is_hit = True
            rec.t = t
            rec.pos = ray.at(t)
            rec.mat = self.material

            outward_normal = Vec3(0)
            if ti.static(self.mix_norm):
                outward_normal = n1 * (1 - u - v) + n2 * u + n3 * v
            else:
                outward_normal = E1.cross(E2)
            rec.set_face_normal(ray, outward_normal.normalized())
        return rec

    @ti.func
    def hit(self, ray: Ray3, tmin: float) -> HitRecord:
        # TODO: Traverse the BVH to find intersection

        rec = HitRecord()
        closest_t = tmin
        curr = self.bvh.nNodes - 1

        while curr >= 0:
            if self.bvh.hitAABB(curr, ray, closest_t):
                if self.bvh.index[curr] >= 0:
                    rec = self.hit_triangle(curr, ray, closest_t)
                    if rec.is_hit:
                        closest_t = min(closest_t, rec.t)
                curr = self.bvh.hitNext[curr]
            else:
                curr = self.bvh.missNext[curr]
        return rec
