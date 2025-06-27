"""
Assignment 3. Fragment Shaders
"""

import taichi as ti
import cv2
from enum import Enum

from mesh import Mesh
from texture import Texture
from camera import Camera
from renderer import Renderer
import transform


class Key(Enum):
    """
    Keyboard maps
    """

    ESC = 27
    SPACE = 32
    TAB = 9
    A = 97
    D = 100
    CLOSE = -1
    NUM0 = 48
    NUM1 = 49
    NUM2 = 50
    NUM3 = 51


def manual_model_transform(
    renderer: Renderer,
    angles=(0, 0, 0),
    scales=(1, 1, 1),
    translates=(0, 0, 0),
):
    renderer.model = transform.get_model_transform(angles, scales, translates)
    renderer.update_mvp_transform()


if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(__file__))
    ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

    ## load mesh & texture ##

    obj_file = "./models/spot/spot_triangulated.obj"
    texture_file = "./models/spot/spot_texture.png"
    # obj_file = "./models/bunny/bunny.obj"
    # texture_file = ""

    mesh = Mesh.from_file(obj_file)
    texture = Texture.from_file(texture_file)
    # print(mesh)
    # print(texture)

    ## renderer ##

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
    renderer.set_mesh(mesh)
    renderer.set_texture(texture)
    renderer.set_MSAA(n=2)

    manual_model_transform(
        renderer, angles=(0, -140, 0), scales=(2.5, 2.5, 2.5), translates=(0, 0, 0)
    )

    renderer.render()
    frame_array = renderer.get_frame_array(cv2=True)

    while True:
        cv2.imshow("Fragment Shaders", frame_array)

        k = cv2.waitKey(0)
        # print(k)

        rerender_flag = False
        match k:
            case Key.ESC.value | Key.CLOSE.value:
                break
            case Key.SPACE.value:
                use_MSAA = False
                rerender_flag = True
                print("reset")
            case Key.TAB.value:
                use_MSAA = True
                rerender_flag = True
                print("use MSAA")

        if rerender_flag:
            renderer.render(use_MSAA)
            frame_array = renderer.get_frame_array(cv2=True)

    cv2.destroyAllWindows()
