# Assignment 1: Draw triangle wireframe

import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32)


def normalize(x, eps=1e-9):
    x_norm = np.linalg.norm(np.array(x))
    if x_norm < eps:
        x_norm += eps
    return x / x_norm


# 视口变换
def get_viewport_matrix(width: float, height: float):
    viewport = tm.mat4(
        [
            [width / 2, 0, 0, (width - 1) / 2],
            [0, height / 2, 0, (height - 1) / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    return viewport


# 相机变换（camera/view）
def get_view_matrix(eye_pos, look_at, eye_up):
    w = -normalize(look_at)
    u = normalize(np.cross(eye_up, w))
    v = np.cross(w, u)
    view = tm.mat4(tm.vec4(u, 0), tm.vec4(v, 0), tm.vec4(w, 0), tm.vec4(0, 0, 0, 1))
    translate = tm.mat4(
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
def get_model_matrix(rotation_angle: float, rotate_axis: str = "z"):
    #  TODO: Implement this function
    #  Create the model matrix for rotating the triangle around the Z axis.
    #  Then return it.

    model = tm.mat4(np.eye(4))

    theta = rotation_angle / 180 * tm.pi
    sin_theta = tm.sin(theta)
    cos_theta = tm.cos(theta)

    if rotate_axis == "z":
        model = tm.mat4(
            [
                [cos_theta, -sin_theta, 0, 0],
                [sin_theta, cos_theta, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
    elif rotate_axis == "y":
        model = tm.mat4(
            [
                [cos_theta, 0, -sin_theta, 0],
                [0, 1, 0, 0],
                [sin_theta, 0, cos_theta, 0],
                [0, 0, 0, 1],
            ]
        )
    elif rotate_axis == "x":
        model = tm.mat4(
            [
                [1, 0, 0, 0],
                [0, cos_theta, -sin_theta, 0],
                [0, sin_theta, cos_theta, 0],
                [0, 0, 0, 1],
            ]
        )
    else:
        raise RuntimeError("Unsupport rotate axis!")

    return model


# 正交变换
def get_orthographic_matrix(
    eye_fov: float, aspect_ratio: float, zNear: float, zFar: float
):
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

    scale = tm.mat4(
        [
            [2 / (r - l), 0, 0, 0],
            [0, 2 / (t - b), 0, 0],
            [0, 0, 2 / (n - f), 0],
            [0, 0, 0, 1],
        ]
    )
    translate = tm.mat4(
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
def get_projection_matrix(
    eye_fov: float, aspect_ratio: float, zNear: float, zFar: float
):
    # TODO: Implement this function
    # Create the projection matrix for the given parameters.
    # Then return it.

    ortho = get_orthographic_matrix(eye_fov, aspect_ratio, zNear, zFar)

    # perspect-to-orthographic
    n, f = -zNear, -zFar
    p2o = tm.mat4(
        [
            [n, 0, 0, 0],
            [0, n, 0, 0],
            [0, 0, n + f, -f * n],
            [0, 0, 1, 0],
        ]
    )

    projection = ortho @ p2o
    return projection


# ==============================================


@ti.kernel
def render(angle: float):
    frame_buf.fill(background_color)  # set background color

    model = get_model_matrix(angle, rotate_axis="z")
    mvp = viewport @ projection @ view @ model

    # print("viewport:", viewport)
    # print("projection:", projection)
    # print("view:", view)
    # print("model:", model)
    # print("mvp:", mvp)

    for i in indices:
        i1, i2, i3 = indices[i]
        v1, v2, v3 = (
            tm.vec4(vertices[i1], 1),
            tm.vec4(vertices[i2], 1),
            tm.vec4(vertices[i3], 1),
        )

        v1 = mvp @ v1
        v2 = mvp @ v2
        v3 = mvp @ v3

        v1 /= v1.w
        v2 /= v2.w
        v3 /= v3.w

        # set_pixel(v1.x, v1.y, line_color)
        # set_pixel(v2.x, v2.y, line_color)
        # set_pixel(v3.x, v3.y, line_color)

        rasterize_wireframe(v1, v2, v3)


@ti.kernel
def update_model_transform(rotation_angle: float):
    model = get_model_matrix(rotation_angle, rotate_axis="z")
    return model


@ti.func
def rasterize_wireframe(v1: ti.template(), v2: ti.template(), v3: ti.template()):
    draw_line(v1.x, v1.y, v2.x, v2.y, line_color)
    draw_line(v2.x, v2.y, v3.x, v3.y, line_color)
    draw_line(v3.x, v3.y, v1.x, v1.y, line_color)


# Bresenham's line drawing algorithm
# reference: https://github.com/miloyip/line/blob/master/line_bresenham.c
@ti.func
def draw_line(x0: int, y0: int, x1: int, y1: int, line_color: ti.template()):
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = int(tm.sign(x1 - x0 + 1e-12))  # sx = 1 if x0 < x1 else -1
    sy = int(tm.sign(y1 - y0 + 1e-12))  # sy = 1 if y0 < y1 else -1
    err = dx / 2
    if dx <= dy:
        err = -dy / 2

    x, y = x0, y0
    while x != x1 or y != y1:
        set_pixel(x, y, line_color)
        e2 = err
        if e2 > -dx:
            err -= dy
            x += sx
        if e2 < dy:
            err += dx
            y += sy


@ti.func
def set_pixel(x: int, y: int, color: ti.template()):
    if x >= 0 and x < width and y >= 0 and y < height:
        frame_buf[x, y] = color


# ================================================

if __name__ == "__main__":
    # define data
    pos = np.array([[2, 0, -2], [0, 2, -2], [-2, 0, -2]], dtype=np.float32)
    ind = np.array([[0, 1, 2]], dtype=np.int32)

    angle = 0

    # camera
    eye_pos = (0, 0, 5)
    eye_up = (0, 1, 0)
    eye_fov = 60
    eye_look_at = (0, 1, -2)
    zNear = 0.1
    zFar = 50

    # colors
    line_color = tm.vec3(1, 1, 1)
    background_color = tm.vec3(0, 0, 0)

    # define display
    width = 1024
    height = 1024
    resolution = (width, height)
    aspect_ratio = 1.0 * height / width

    # taichi data
    vertices = ti.Vector.field(3, float, shape=len(pos))
    indices = ti.Vector.field(3, int, shape=len(ind))

    vertices.from_numpy(pos)
    indices.from_numpy(ind)

    frame_buf = ti.Vector.field(3, float, shape=resolution)  # 屏幕像素颜色信息（rbg）

    # transform matrix
    viewport = get_viewport_matrix(width, height)
    view = get_view_matrix(eye_pos, eye_look_at, eye_up)
    projection = get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar)
    model = get_model_matrix(angle, rotate_axis="z")

    mvp = viewport @ projection @ view @ model

    print("viewport:", viewport)
    print("projection:", projection)
    print("view:", view)
    print("model:", model)
    print("mvp:", mvp)

    # rendering
    window = ti.ui.Window("draw triangle wireframe", resolution)
    canvas = window.get_canvas()

    render(angle)

    rotate_delta = 1  # 定义旋转速度
    while window.running:
        if window.is_pressed(ti.ui.ESCAPE):  # 按ESC退出
            break

        if window.is_pressed("a"):  # 按 A 键绕z轴逆时针旋转
            angle = (angle + rotate_delta) % 360
            render(angle)

        if window.is_pressed("d"):  # 按 D 键绕z轴顺时针旋转
            angle = (angle - rotate_delta) % 360
            render(angle)
            
        canvas.set_image(frame_buf)
        window.show()
