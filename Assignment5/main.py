# Assignment 5. Whitted-style RayTracer


import taichi as ti
import taichi.math as tm

from models import *
from renderer import *

ti.init(arch=ti.gpu)


# Variables
WIDTH = 1280
HEIGHT = 960
background_color = (0.235294, 0.67451, 0.843137)
MAX_DEPTH = 10
MAX_STACK_SIZE = 50


renderer = Renderer(
    WIDTH,
    HEIGHT,
    background_color=background_color,
    max_depth=MAX_DEPTH,
    max_stack_size=MAX_STACK_SIZE,
)


def init_scene():
    sph1 = Sphere((-1, 0, -12), 2)
    sph1.set_material(m_type=DIFFUSE_AND_GLOSSY, diffuse_color=(0.6, 0.7, 0.8))

    sph2 = Sphere((0.5, -0.5, -8), 1.5)
    sph2.set_material(m_type=REFLECTION_AND_REFRACTION, ior=1.5)

    mesh = MeshTriangle(
        vertices=[[-5, -3, -6], [5, -3, -6], [5, -3, -16], [-5, -3, -16]],
        indices=[[0, 1, 3], [1, 2, 3]],
        st_coords=[[0, 0], [1, 0], [1, 1], [0, 1]],
    )
    mesh.set_material(m_type=DIFFUSE_AND_GLOSSY)

    renderer.world.add(sph1)
    renderer.world.add(sph2)
    renderer.world.add(mesh)

    renderer.lights.add_ambient_light(10)
    renderer.lights.add_point_light((-20, 70, 20), 0.5)
    renderer.lights.add_point_light((30, 50, -12), 0.5)


if __name__ == "__main__":
    init_scene()

    window = ti.ui.Window("Whitted-Style Raytracer", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    renderer.render()

    while window.running:
        if window.is_pressed(ti.ui.ESCAPE):
            break

        canvas.set_image(renderer.frame_buf)
        window.show()
