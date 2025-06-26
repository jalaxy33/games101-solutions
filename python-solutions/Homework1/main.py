"""
Homework 1. Draw triangle wireframe
"""

import numpy as np
import taichi as ti
import cv2
from enum import Enum


from camera import Camera
from renderer import Renderer
import transform


class Key(Enum):
    """
    Keyboard maps
    """

    ESC = 27
    SPACE = 32
    A = 97
    D = 100
    CLOSE = -1


#  TODO: Implement this function
#  Create the model matrix for rotating the triangle around the Z axis.
#  Then return it.
def rotate_around_z(renderer: Renderer, angle: float):
    """
    rotate along z axis
    """
    model = transform.get_model_transform(angles=(0, 0, angle))
    renderer.model = model
    renderer.update_mvp_transform()


if __name__ == "__main__":
    ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

    # define triangles
    vertices = np.array([[2, 0, -2], [0, 2, -2], [-2, 0, -2]], dtype=np.float32)
    indices = np.array([[0, 1, 2]], dtype=np.int32)

    camera = Camera(
        eye_pos=(0, 0, 5),
        vup=(0, 1, 0),
        look_at=(0, 1, -2),
        vfov=60,
        zNear=0.1,
        zFar=50,
    )

    renderer = Renderer(
        width=1024, height=1024, line_color=(1, 1, 1), background_color=(0, 0, 0)
    )
    renderer.set_camera(camera)
    renderer.set_mesh(vertices, indices)

    renderer.render()
    frame_array = renderer.get_frame_array(cv2=True)

    angle = 0
    rotate_delta = 10  # rotation speed
    while True:
        cv2.imshow("Triangle Wireframe", frame_array)

        k = cv2.waitKey(0)
        # print(k)

        rerender_flag = False
        match k:
            case Key.ESC.value | Key.CLOSE.value:
                break
            case Key.A.value:
                angle = (angle + rotate_delta) % 360
                rotate_around_z(renderer, angle)
                rerender_flag = True
            case Key.D.value:
                angle = (angle - rotate_delta) % 360
                rotate_around_z(renderer, angle)
                rerender_flag = True
            case Key.SPACE.value:
                angle = 0
                rotate_around_z(renderer, angle)
                rerender_flag = True

        if rerender_flag:
            renderer.render()
            frame_array = renderer.get_frame_array(cv2=True)

    cv2.destroyAllWindows()
