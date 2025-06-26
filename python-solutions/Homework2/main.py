"""
Homework 2. Draw 2 overlapping triangles & Implement Anti-aliasing.
"""

import numpy as np
import taichi as ti
import cv2
from enum import Enum


from camera import Camera
from renderer import Renderer


class Key(Enum):
    """
    Keyboard maps
    """

    ESC = 27
    SPACE = 32
    A = 97
    D = 100
    CLOSE = -1
    NUM0 = 48
    NUM1 = 49
    NUM2 = 50
    NUM3 = 51


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(__file__))

    ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

    ## define data

    vertices = np.array(
        [
            (2, 0, -2),
            (0, 2, -2),
            (-2, 0, -2),
            (3.5, -1, -5),
            (2.5, 1.5, -5),
            (-1, 0.5, -5),
        ],
        dtype=np.float32,
    )
    indices = np.array(
        [
            (0, 1, 2),
            (3, 4, 5),
        ],
        dtype=np.int32,
    )
    vertex_colors = np.array(
        [
            (217.0, 238.0, 185.0),
            (217.0, 238.0, 185.0),
            (217.0, 238.0, 185.0),
            (185.0, 217.0, 238.0),
            (185.0, 217.0, 238.0),
            (185.0, 217.0, 238.0),
        ],
        dtype=np.float32,
    )
    vertex_colors = vertex_colors / 255.0

    assert len(vertices) == len(vertex_colors)

    ## define renderer

    camera = Camera(
        eye_pos=(0, 0, 5),
        vup=(0, 1, 0),
        look_at=(0, 0, -5),
        vfov=60,
        zNear=0.1,
        zFar=50,
    )

    renderer = Renderer(width=1024, height=1024, background_color=(0, 0, 0))
    renderer.set_camera(camera)
    renderer.set_mesh(vertices, indices, vertex_colors)
    renderer.set_MSAA(n=2)

    renderer.render()
    frame_array = renderer.get_frame_array(cv2=True)

    while True:
        cv2.imshow("Two triangles", frame_array)

        k = cv2.waitKey(0)
        # print(k)

        rerender_flag = False
        match k:
            case Key.ESC.value | Key.CLOSE.value:
                break
            case Key.SPACE.value:
                use_MSAA = False
                rerender_flag = True
                print("no MSAA")
            case Key.NUM1.value | Key.NUM2.value | Key.NUM3.value:
                use_MSAA = True
                MSAA_N = 2 ** (k - Key.NUM0.value)
                renderer.set_MSAA(MSAA_N)
                rerender_flag = True
                print(f"MSAA_N = {MSAA_N}")

        if rerender_flag:
            renderer.render(use_MSAA)
            frame_array = renderer.get_frame_array(cv2=True)

    cv2.destroyAllWindows()
