import taichi as ti

from common import *


@ti.pyfunc
def eye4() -> Mat4x4:
    return Mat4x4(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


@ti.pyfunc
def normalize(v):
    return v / v.norm()


@ti.pyfunc
def get_viewport_matrix(width: float, height: float) -> Mat4x4:
    """视口变换"""

    viewport = Mat4x4(
        [
            [width / 2, 0, 0, (width - 1) / 2],
            [0, height / 2, 0, (height - 1) / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return viewport


@ti.pyfunc
def get_view_matrix(eye_pos, look_at, vup) -> Mat4x4:
    """相机变换（camera/view）"""

    w = -normalize(look_at)
    u = normalize(vup.cross(w))
    v = w.cross(u)
    view = Mat4x4(Vec4(u, 0), Vec4(v, 0), Vec4(w, 0), Vec4(0, 0, 0, 1))
    translate = Mat4x4(
        [
            [1, 0, 0, -eye_pos[0]],
            [0, 1, 0, -eye_pos[1]],
            [0, 0, 1, -eye_pos[2]],
            [0, 0, 0, 1],
        ]
    )
    view = view @ translate
    return view


@ti.pyfunc
def get_model_matrix(angles=(0, 0, 0), scales=(1, 1, 1), translates=(0, 0, 0)) -> Mat4x4:
    """模型变换"""

    # rotation
    theta_x = angles[0] / 180.0 * PI
    theta_y = angles[1] / 180.0 * PI
    theta_z = angles[2] / 180.0 * PI

    sin_x, cos_x = ti.sin(theta_x), ti.cos(theta_x)
    sin_y, cos_y = ti.sin(theta_y), ti.cos(theta_y)
    sin_z, cos_z = ti.sin(theta_z), ti.cos(theta_z)

    rotate_x = Mat4x4(
        [
            [cos_x, -sin_x, 0, 0],
            [sin_x, cos_x, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    rotate_y = Mat4x4(
        [
            [cos_y, 0, -sin_y, 0],
            [0, 1, 0, 0],
            [sin_y, 0, cos_y, 0],
            [0, 0, 0, 1],
        ]
    )
    rotate_z = Mat4x4(
        [
            [1, 0, 0, 0],
            [0, cos_z, -sin_z, 0],
            [0, sin_z, cos_z, 0],
            [0, 0, 0, 1],
        ]
    )

    rotation = rotate_x @ rotate_y @ rotate_z

    # scale
    scale = Mat4x4(
        [
            [scales[0], 0, 0, 0],
            [0, scales[0], 0, 0],
            [0, 0, scales[0], 0],
            [0, 0, 0, 1],
        ]
    )

    # translate
    translate = Mat4x4(
        [
            [1, 0, 0, translates[0]],
            [0, 1, 0, translates[1]],
            [0, 0, 1, translates[2]],
            [0, 0, 0, 1],
        ]
    )
    model = translate @ scale @ rotation
    return model


@ti.pyfunc
def get_orthographic_matrix(vfov: float, aspect_ratio: float, zNear: float, zFar: float) -> Mat4x4:
    """正交变换"""

    # display area
    # near-far
    n = -zNear
    f = -zFar
    # top-bottom
    alpha = vfov / 180.0 * PI
    t = ti.tan(alpha / 2) * abs(n)
    b = -t
    # right-left
    r = t * aspect_ratio
    l = -r

    ortho = Mat4x4(
        [
            [2 / (r - l), 0, 0, -(r + l) / (r - l)],
            [0, 2 / (t - b), 0, -(t + b) / (t - b)],
            [0, 0, 2 / (n - f), -(n + f) / (n - f)],
            [0, 0, 0, 1],
        ]
    )

    return ortho


@ti.pyfunc
def get_projection_matrix(eye_fov: float, aspect_ratio: float, zNear: float, zFar: float) -> Mat4x4:
    """投影变换"""

    ortho = get_orthographic_matrix(eye_fov, aspect_ratio, zNear, zFar)

    # perspect-to-orthographic
    n, f = -zNear, -zFar
    p2o = Mat4x4(
        [
            [n, 0, 0, 0],
            [0, n, 0, 0],
            [0, 0, n + f, -f * n],
            [0, 0, 1, 0],
        ]
    )

    projection = ortho @ p2o
    return projection
