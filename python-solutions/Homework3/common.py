import numpy as np
import taichi as ti

NpArr = np.ndarray
INF = np.inf

Vec2 = ti.types.vector(2, ti.float32)
Vec3 = ti.types.vector(3, ti.float32)
Vec4 = ti.types.vector(4, ti.float32)


def generate_colors() -> NpArr:
    return np.random.randint(0, 256, size=3).astype(np.float32) / 255.0
