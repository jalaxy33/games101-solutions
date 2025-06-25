import numpy as np
import taichi as ti


Vec3 = ti.types.vector(3, float)
Vec4 = ti.types.vector(4, float)
Mat4x4 = ti.types.matrix(4, 4, float)
NpArr = np.ndarray
PI = ti.math.pi
INF = ti.math.inf
