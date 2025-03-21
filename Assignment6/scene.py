import taichi as ti
import taichi.math as tm

from common import *


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

    @ti.func
    def hit(self, ray: Ray3, tmin=Inf) -> HitRecord:
        rec = HitRecord()
        closest_so_far = tmin

        for i in ti.static(range(len(self.objs))):
            temp_rec = self.objs[i].hit(ray, closest_so_far)
            if temp_rec.is_hit and temp_rec.t < closest_so_far:
                closest_so_far = temp_rec.t
                rec = temp_rec

        return rec


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
