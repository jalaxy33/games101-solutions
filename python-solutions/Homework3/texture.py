import os
import numpy as np
import taichi as ti
import open3d as o3d
from dataclasses import dataclass, field
from typing import Self

from common import NpArr, Vec3


@dataclass
class Texture:
    image: NpArr = field(default_factory=np.empty((0, 0, 3), np.float32))
    default_color: tuple[float, float, float] = field(default=(0, 0, 0))

    @classmethod
    def from_file(cls, tex_file: str) -> Self:
        """Load texture from file"""
        return load_texture(tex_file)

    def __post_init__(self):
        self.W = self.image.shape[0]
        self.H = self.image.shape[1]
        self.set_default_color(self.default_color)

        if len(self.image):
            self.image_gpu = ti.Vector.field(3, ti.float32)
            ti.root.dense(ti.ij, (self.W, self.H)).place(self.image_gpu)
            self.image_gpu.from_numpy(self.image)

    def set_default_color(self, color: tuple[float, float, float]):
        self.default_color = color
        self._default_color = Vec3(color)

    def get_default_color(self):
        return self._default_color

    @ti.pyfunc
    def get_color(self, u: float, v: float) -> Vec3:
        i, j = int(u * self.W), int(v * self.H)
        color = self.get_default_color()
        if i >= 0 and i < self.W and j >= 0 and j < self.H:
            color = self.image[i, j]
        return color


def load_texture(tex_file: str) -> Texture:
    os.path.exists(tex_file), f"'{tex_file}' not exist!"

    if tex_file:
        tex_img = np.asarray(o3d.io.read_image(tex_file), np.float32) / 255
    else:
        tex_img = np.empty((0, 0, 3), np.float32)
    return Texture(image=tex_img)
