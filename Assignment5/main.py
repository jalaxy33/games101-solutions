# Assignment 5. Whitted-style Raytracer
# Reference:
#   https://github.com/erizmr/taichi_ray_tracing/blob/master/3_2_whitted_style_ray_tracing.py
#   https://github.com/Jiay1C/path_tracing_obj/blob/master/path_tracing_obj.py
#   https://shao.fun/blog/w/taichi-ray-tracing.html

import os

os.chdir(os.path.dirname(__file__))

import taichi as ti
import taichi.math as tm

from models import *

ti.init(arch=ti.gpu)


# Args
WIDTH = 1280
HEIGHT = 960
background_color = (0.235294, 0.67451, 0.843137)

# Global variables
FrameBuffer = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))


@ti.data_oriented
class Scene:
    def __init__(self):
        self.models = []
        self.lights = []
        self.epsilon = 1e-5

    def add_model(self, model):
        self.models.append(model)

    def clear_models(self):
        self.models.clear()

    def add_light(self, light):
        self.lights.append(light)

    def clear_lights(self):
        self.lights.clear()


scene = Scene()


def init_scene():
    sph1 = Sphere(center=(-1, 0, -12), radius=2)
    sph1.set_material(m_type=DIFFUSE_AND_GLOSSY, diffuse_color=(0.6, 0.7, 0.8))

    sph2 = Sphere(center=(0.5, -0.5, -8), radius=1.5)
    sph2.set_material(m_type=REFLECTION_AND_REFRACTION, refrac_idx=1.5)

    scene.add_model(sph1)
    scene.add_model(sph2)


if __name__ == "__main__":
    init_scene()

    window = ti.ui.Window("Whitted-style Raytracer", (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    FrameBuffer.fill(background_color)

    while window.running:
        if window.is_pressed(ti.ui.ESCAPE):  # Press <ESC> to exit
            break

        canvas.set_image(FrameBuffer)
        window.show()
