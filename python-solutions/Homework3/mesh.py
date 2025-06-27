import os
import numpy as np
import taichi as ti
import open3d as o3d
from dataclasses import dataclass, field
from typing import Self

from common import NpArr, generate_colors
from triangle import Triangle


@dataclass
class Mesh:
    vertices: NpArr
    indices: NpArr
    normals: NpArr = field(default_factory=lambda: np.empty((0, 3), np.float32))
    vertex_colors: NpArr = field(default_factory=lambda: np.empty((0, 3), np.float32))
    triangle_uvs: NpArr = field(default_factory=lambda: np.empty((0, 2), np.float32))

    @classmethod
    def from_file(cls, mesh_file: str) -> Self:
        """Load Mesh from file"""
        return load_mesh(mesh_file)

    def __post_init__(self):
        self.vertices_4d = np.hstack(
            [self.vertices, np.ones((len(self.vertices), 1))], dtype=np.float32
        )

        if len(self.vertex_colors):
            self.vertex_colors /= self.vertex_colors
        else:
            self.vertex_colors = np.zeros((len(self.vertices), 3), np.float32)
            self.vertex_colors[:] = generate_colors()

        self.verts_gpu = ti.Vector.field(4, ti.float32)
        self.inds_gpu = ti.Vector.field(3, ti.int32)
        self.norms_gpu = ti.Vector.field(3, ti.float32)
        self.colors_gpu = ti.Vector.field(3, ti.float32)
        self.uvs_gpu = ti.Vector.field(2, ti.float32)

        ti.root.dense(ti.i, len(self.vertices)).place(self.verts_gpu)
        ti.root.dense(ti.i, len(self.indices)).place(self.inds_gpu)
        ti.root.dense(ti.i, len(self.normals)).place(self.norms_gpu)
        ti.root.dense(ti.i, len(self.vertex_colors)).place(self.colors_gpu)
        ti.root.dense(ti.i, len(self.indices) * 3).place(self.uvs_gpu)

        self.verts_gpu.from_numpy(self.vertices_4d)
        self.inds_gpu.from_numpy(self.indices)
        self.norms_gpu.from_numpy(self.normals)
        self.colors_gpu.from_numpy(self.vertex_colors)

        if len(self.triangle_uvs):
            self.uvs_gpu.from_numpy(self.triangle_uvs)

    @ti.pyfunc
    def get_gpu_triangle(self, i: int) -> Triangle:
        i1, i2, i3 = self.inds_gpu[i]
        v1, v2, v3 = self.verts_gpu[i1], self.verts_gpu[i2], self.verts_gpu[i3]
        n1, n2, n3 = self.norms_gpu[i1], self.norms_gpu[i2], self.norms_gpu[i3]
        c1, c2, c3 = self.colors_gpu[i1], self.colors_gpu[i2], self.colors_gpu[i3]

        v1 /= v1.w
        v2 /= v2.w
        v3 /= v3.w

        _I = 3 * i
        uv1, uv2, uv3 = self.uvs_gpu[_I], self.uvs_gpu[_I + 1], self.uvs_gpu[_I + 2]

        t = Triangle(v1, v2, v3, n1, n2, n3, c1, c2, c3, uv1, uv2, uv3)
        return t


def load_mesh(mesh_file: str) -> Mesh:
    assert os.path.exists(mesh_file), f"'{mesh_file}' not exist!"

    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, np.float32)
    indices = np.asarray(mesh.triangles, np.int32)
    normals = np.asarray(mesh.vertex_normals, np.float32)
    vertex_colors = np.asarray(mesh.vertex_colors, np.float32)
    triangle_uvs = np.asarray(mesh.triangle_uvs, np.float32)

    return Mesh(
        vertices=vertices,
        indices=indices,
        normals=normals,
        vertex_colors=vertex_colors,
        triangle_uvs=triangle_uvs,
    )
