import taichi as ti

from common import *
from Material import *


@ti.dataclass
class AmbientLight:
    Ia: Vec3


@ti.dataclass
class PointLight:
    pos: Vec3  # position
    I: Vec3  # Intensity


@ti.data_oriented
class LightList:
    def __init__(self):
        self.ambient_light = AmbientLight()
        self.point_lights = []

    def add_ambient_light(self, intensity: Vec3):
        self.ambient_light.Ia = Vec3(intensity)

    def add_point_light(self, pos: Vec3, intensity: Vec3):
        self.point_lights.append(PointLight(Vec3(pos), Vec3(intensity)))


@ti.data_oriented
class HittableList:
    def __init__(self):
        self.objs = []

    def clear(self):
        self.objs.clear()

    def add(self, obj):
        self.objs.append(obj)

    @ti.pyfunc
    def hit(self, origin, direction, t_near=Inf):
        direction = direction.normalized()

        is_hit = False
        closest_t = t_near
        hit_pos = Vec3(Inf)
        hit_norm = Vec3(0)
        hit_mat = Material()

        for i in ti.static(range(len(self.objs))):
            _hit, _t, _pos, _n, _mat = self.objs[i].hit(origin, direction, t_near)
            if _hit and _t < closest_t:
                is_hit = _hit
                closest_t = _t
                hit_pos = _pos
                hit_norm = _n
                hit_mat = _mat

        return is_hit, closest_t, hit_pos, hit_norm, hit_mat


@ti.data_oriented
class Scene:
    def __init__(self):
        self.lights = LightList()
        self.world = HittableList()

    def add_obj(self, obj):
        self.world.add(obj)

    def add_ambient_light(self, intensity: Vec3):
        self.lights.add_ambient_light(intensity)

    def add_point_light(self, pos: Vec3, intensity: Vec3):
        self.lights.add_point_light(pos, intensity)
