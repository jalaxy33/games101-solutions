import numpy as np
import taichi as ti
import taichi.math as tm


vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)
mat4 = ti.types.matrix(4, 4, float)


def normalize(x, eps=1e-9):
    x_norm = np.linalg.norm(np.array(x))
    if x_norm < eps:
        x_norm += eps
    return x / x_norm


# 视口变换
def get_viewport_matrix(width: float, height: float) -> mat4:
    viewport = mat4(
        [
            [width / 2, 0, 0, (width - 1) / 2],
            [0, height / 2, 0, (height - 1) / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return viewport


# 相机变换（camera/view）
def get_view_matrix(eye_pos, look_at, eye_up) -> mat4:
    w = -normalize(eye_pos - look_at)
    u = normalize(np.cross(eye_up, w))
    v = np.cross(w, u)
    view = mat4(vec4(u, 0), vec4(v, 0), vec4(w, 0), vec4(0, 0, 0, 1))
    translate = mat4(
        [
            [1, 0, 0, -eye_pos[0]],
            [0, 1, 0, -eye_pos[1]],
            [0, 0, 1, -eye_pos[2]],
            [0, 0, 0, 1],
        ]
    )
    view = view @ translate
    return view


# 模型变换
def get_model_matrix(
    angles=(0, 0, 0),
    scales=(1, 1, 1),
    translates=(0, 0, 0),
) -> mat4:
    model = mat4(np.eye(4))

    # rotation
    theta_x = angles[0] / 180 * tm.pi
    theta_y = angles[1] / 180 * tm.pi
    theta_z = angles[2] / 180 * tm.pi

    sin_x, cos_x = tm.sin(theta_x), tm.cos(theta_x)
    sin_y, cos_y = tm.sin(theta_y), tm.cos(theta_y)
    sin_z, cos_z = tm.sin(theta_z), tm.cos(theta_z)

    rotate_x = mat4(
        [
            [cos_x, -sin_x, 0, 0],
            [sin_x, cos_x, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    rotate_y = mat4(
        [
            [cos_y, 0, -sin_y, 0],
            [0, 1, 0, 0],
            [sin_y, 0, cos_y, 0],
            [0, 0, 0, 1],
        ]
    )
    rotate_z = mat4(
        [
            [1, 0, 0, 0],
            [0, cos_z, -sin_z, 0],
            [0, sin_z, cos_z, 0],
            [0, 0, 0, 1],
        ]
    )

    rotation = rotate_x @ rotate_y @ rotate_z

    # scale
    scale = mat4(
        [
            [scales[0], 0, 0, 0],
            [0, scales[0], 0, 0],
            [0, 0, scales[0], 0],
            [0, 0, 0, 1],
        ]
    )

    # translate
    translate = mat4(
        [
            [1, 0, 0, translates[0]],
            [0, 1, 0, translates[1]],
            [0, 0, 1, translates[2]],
            [0, 0, 0, 1],
        ]
    )
    model = translate @ scale @ rotation
    return model


# 正交变换
def get_orthographic_matrix(eye_fov: float, aspect_ratio: float, zNear: float, zFar: float) -> mat4:
    # display area
    # near-far
    n = -zNear
    f = -zFar
    # top-bottom
    alpha = eye_fov / 180 * tm.pi
    t = tm.tan(alpha / 2) * abs(n)
    b = -t
    # right-left
    r = t * aspect_ratio
    l = -r

    scale = mat4(
        [
            [2 / (r - l), 0, 0, 0],
            [0, 2 / (t - b), 0, 0],
            [0, 0, 2 / (n - f), 0],
            [0, 0, 0, 1],
        ]
    )
    translate = mat4(
        [
            [1, 0, 0, -(r + l) / 2],
            [0, 1, 0, -(t + b) / 2],
            [0, 0, 1, -(n + f) / 2],
            [0, 0, 0, 1],
        ]
    )
    ortho = scale @ translate
    return ortho


# 投影变换
def get_projection_matrix(eye_fov: float, aspect_ratio: float, zNear: float, zFar: float) -> mat4:
    ortho = get_orthographic_matrix(eye_fov, aspect_ratio, zNear, zFar)

    # perspect-to-orthographic
    n, f = -zNear, -zFar
    p2o = mat4(
        [
            [n, 0, 0, 0],
            [0, n, 0, 0],
            [0, 0, n + f, -f * n],
            [0, 0, 1, 0],
        ]
    )

    projection = ortho @ p2o
    return projection
