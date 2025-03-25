import taichi as ti
import taichi.math as tm

# Material Types
DIFFUSE_AND_GLOSSY = 0
REFLECTION_AND_REFRACTION = 1
REFLECTION = 2


@ti.dataclass
class Material:
    m_type: int  # 材质类型
    m_color: tm.vec3
    m_emission: tm.vec3
    ior: float  # 折射系数 refract index
    kd: tm.vec3  # 漫反射系数
    ks: tm.vec3  # 镜面反射系数
    ka: tm.vec3  # 环境光系数
    spec_exp: float  # 高光系数
