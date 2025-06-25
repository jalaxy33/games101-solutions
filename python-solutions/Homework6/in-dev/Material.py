import taichi as ti
from common import *

# Material Types
DIFFUSE_AND_GLOSSY = 0
REFLECTION_AND_REFRACTION = 1
REFLECTION = 2


@ti.dataclass
class Material:
    m_type: int  # 材质类型
    m_color: Vec3
    m_emission: Vec3
    ior: float  # 折射系数 refract index
    kd: Vec3  # 漫反射系数
    ks: Vec3  # 镜面反射系数
    ka: Vec3  # 环境光系数
    spec_exp: float  # 高光系数
