# Assignment 6.

import os

os.chdir(os.path.dirname(__file__))


import taichi as ti
import taichi.math as tm
from vispy.io import read_mesh

from common import *
from model import *
from scene import *
from renderer import *

ti.init(arch=ti.gpu)


# Variables
WIDTH = 1280
HEIGHT = 960
background_color = (0.235294, 0.67451, 0.843137)
MAX_DEPTH = 10
MAX_STACK_SIZE = 50
VFOV = 100
EYE_POS = (-1, 5, 10)


renderer = Renderer(
    WIDTH,
    HEIGHT,
    background_color=background_color,
    max_depth=MAX_DEPTH,
    max_stack_size=MAX_STACK_SIZE,
    vfov=VFOV,
    eye_pos=EYE_POS,
)


def init_bunny_scene():
    bunny_obj = "e:/projects/games101/Assignment6/task/Assignment6/models/bunny/bunny.obj"
    vertices, indices, normals, texcoords = read_mesh(bunny_obj)
    vertices = vertices * 60.0

    bunny = BVHMesh(vertices, indices, normals)
    bunny.set_material(m_type=DIFFUSE_AND_GLOSSY, m_color=(0.5, 0.5, 0.5), kd=0.6, ks=0, spec_exp=0)
    bunny.build_bvh(split_method=SplitMethod.SAH)

    scene = Scene()

    scene.add_obj(bunny)

    scene.add_ambient_light(10)
    scene.add_point_light((-20, 70, 20), 1)
    scene.add_point_light((20, 70, 20), 1)

    renderer.set_scene(scene)


if __name__ == "__main__":
    init_bunny_scene()

    window = ti.ui.Window("BVH + Whitted-style raytracing", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    renderer.render()

    while window.running and not window.is_pressed(ti.ui.ESCAPE):
        canvas.set_image(renderer.frame_buf)
        window.show()
