import numpy as np

from common import NpArr


def normalize(x: NpArr, eps=1e-9) -> NpArr:
    return x / (np.linalg.norm(x) + eps)


def get_viewport_transform(width: float, height: float) -> NpArr:
    """
    视口变换
    """
    viewport = np.array(
        [
            [width / 2, 0, 0, (width - 1) / 2],
            [0, height / 2, 0, (height - 1) / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return viewport


def get_view_transform(eye_pos: NpArr, look_at: NpArr, vup: NpArr) -> NpArr:
    """
    相机变换 (camera/view)
    """
    w = -normalize(look_at)
    u = normalize(np.cross(vup, w))
    v = np.cross(w, u)

    view = np.vstack(
        [
            np.hstack([u, [0]]),
            np.hstack([v, [0]]),
            np.hstack([w, [0]]),
            np.array([0, 0, 0, 1]),
        ]
    )

    translate = np.array(
        [
            [1, 0, 0, -eye_pos[0]],
            [0, 1, 0, -eye_pos[1]],
            [0, 0, 1, -eye_pos[2]],
            [0, 0, 0, 1],
        ]
    )
    view = view @ translate
    return view


#  TODO: Implement this function
#  Create the model matrix for rotating the triangle around the Z axis.
#  Then return it.
def get_model_transform(
    angles=(0, 0, 0),
    scales=(1, 1, 1),
    translates=(0, 0, 0),
) -> NpArr:
    """
    模型变换
    """
    model = np.eye(4)

    # rotation
    theta_x = np.deg2rad(angles[0])
    theta_y = np.deg2rad(angles[1])
    theta_z = np.deg2rad(angles[2])

    sin_x, cos_x = np.sin(theta_x), np.cos(theta_x)
    sin_y, cos_y = np.sin(theta_y), np.cos(theta_y)
    sin_z, cos_z = np.sin(theta_z), np.cos(theta_z)

    rotate_x = np.array(
        [
            [1, 0, 0, 0],
            [0, cos_x, -sin_x, 0],
            [0, sin_x, cos_x, 0],
            [0, 0, 0, 1],
        ]
    )
    rotate_y = np.array(
        [
            [cos_y, 0, -sin_y, 0],
            [0, 1, 0, 0],
            [sin_y, 0, cos_y, 0],
            [0, 0, 0, 1],
        ]
    )
    rotate_z = np.array(
        [
            [cos_z, -sin_z, 0, 0],
            [sin_z, cos_z, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    rotation = rotate_z @ rotate_y @ rotate_x

    # scale
    scale = np.array(
        [
            [scales[0], 0, 0, 0],
            [0, scales[0], 0, 0],
            [0, 0, scales[0], 0],
            [0, 0, 0, 1],
        ]
    )

    # translate
    translate = np.array(
        [
            [1, 0, 0, translates[0]],
            [0, 1, 0, translates[1]],
            [0, 0, 1, translates[2]],
            [0, 0, 0, 1],
        ]
    )
    model = translate @ scale @ rotation
    return model


def get_orthographic_transform(
    vfov: float, aspect_ratio: float, zNear: float, zFar: float
) -> NpArr:
    """
    正交变换
    """
    # display area
    # near-far
    n = -zNear
    f = -zFar
    # top-bottom
    alpha = np.deg2rad(vfov)
    t = np.tan(alpha / 2) * abs(n)
    b = -t
    # right-left
    r = t * aspect_ratio
    l = -r

    scale = np.array(
        [
            [2 / (r - l), 0, 0, 0],
            [0, 2 / (t - b), 0, 0],
            [0, 0, 2 / (n - f), 0],
            [0, 0, 0, 1],
        ]
    )
    translate = np.array(
        [
            [1, 0, 0, -(r + l) / 2],
            [0, 1, 0, -(t + b) / 2],
            [0, 0, 1, -(n + f) / 2],
            [0, 0, 0, 1],
        ]
    )
    ortho = scale @ translate
    return ortho


# TODO: Implement this function
# Create the projection matrix for the given parameters.
# Then return it.
def get_projection_transform(
    vfov: float, aspect_ratio: float, zNear: float, zFar: float
) -> NpArr:
    """
    投影变换
    """
    ortho = get_orthographic_transform(vfov, aspect_ratio, zNear, zFar)

    # perspect-to-orthographic
    n, f = -zNear, -zFar
    p2o = np.array(
        [
            [n, 0, 0, 0],
            [0, n, 0, 0],
            [0, 0, n + f, -f * n],
            [0, 0, 1, 0],
        ]
    )

    projection = ortho @ p2o
    return projection


def get_mvp_transform(
    viewport: NpArr, projection: NpArr, view: NpArr, model: NpArr
) -> NpArr:
    """
    最终变换
    """
    return viewport @ projection @ view @ model
