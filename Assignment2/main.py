import numpy as np
import taichi as ti
import taichi.math as tm


ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32)


@ti.func
def inside_triangle(
    x: float, y: float, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
):
    # TODO: Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    alpha, beta, gamma = compute_barycentric(x, y, x1, y1, x2, y2, x3, y3)
    return alpha >= 0 and beta >= 0 and gamma >= 0  # 在内部或边界上


@ti.func
def rasterize_triangle(
    v1: ti.template(),
    v2: ti.template(),
    v3: ti.template(),
    c1: ti.template(),
    c2: ti.template(),
    c3: ti.template(),
):
    # TODO: Find out the bounding box of current triangle.
    # iterate through the pixel and find if the current pixel is inside the triangle

    # If so, use the following code to get the interpolated z value.
    # auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    # float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    # float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    # z_interpolated *= w_reciprocal;

    # TODO: set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.

    # 创建 bounding box
    x_min = int(tm.floor(tm.min(v1.x, v2.x, v3.x)))
    x_max = int(tm.ceil(tm.max(v1.x, v2.x, v3.x)))
    y_min = int(tm.floor(tm.min(v1.y, v2.y, v3.y)))
    y_max = int(tm.ceil(tm.max(v1.y, v2.y, v3.y)))

    x1, y1, z1, w1 = v1
    x2, y2, z2, w2 = v2
    x3, y3, z3, w3 = v3
    f_alpha = barycentric_ij(x1, y1, x2, y2, x3, y3)
    f_beta = barycentric_ij(x2, y2, x3, y3, x1, y1)
    f_gamma = barycentric_ij(x3, y3, x1, y1, x2, y2)

    # 对于三角形共边的情况
    # 设置一个屏幕外的点，如果该点在某个三角形内，就将边的颜色设为该三角形的颜色
    alpha_check = f_alpha * barycentric_ij(-1, -1, x2, y2, x3, y3) > 0
    beta_check = f_beta * barycentric_ij(-1, -1, x3, y3, x1, y1) > 0
    gamma_check = f_gamma * barycentric_ij(-1, -1, x1, y1, x2, y2) > 0

    if MSAA:
        step_size = 1 / MSAA_N
        start_offset = -0.5 + 1 / (2 * MSAA_N)

        for x, y in ti.ndrange((x_min, x_max + 1), (y_min, y_max + 1)):
            x_start = x + start_offset
            y_start = y + start_offset

            color = tm.vec3(0)  # pixel color
            depth = -tm.inf  # store depth
            for i, j in ti.ndrange(MSAA_N, MSAA_N):
                x_curr = x_start + i * step_size
                y_curr = y_start + j * step_size

                f23 = barycentric_ij(x_curr, y_curr, x2, y2, x3, y3)
                f31 = barycentric_ij(x_curr, y_curr, x3, y3, x1, y1)
                # f12 = barycentric_ij(x_curr, y_curr, x1, y1, x2, y2)

                alpha = f23 / f_alpha
                beta = f31 / f_beta
                # gamma = f12 / f_gamma
                gamma = 1 - alpha - beta

                # 在内部或边界上
                if (
                    (alpha > 0 or (alpha == 0 and alpha_check))
                    and (beta > 0 or (beta == 0 and beta_check))
                    and (gamma > 0 or (gamma == 0 and gamma_check))
                ):
                    c = alpha * c1 + beta * c2 + gamma * c3
                    color += c

                    # 计算插值深度
                    w = alpha * w1 + beta * w2 + gamma * w3
                    z = alpha * z1 / w1 + beta * z2 / w2 + gamma * z3 / w3
                    z /= w

                    if depth < z:
                        depth = z

            color /= MSAA_N * MSAA_N  # 周围超像素的颜色取平均
            if depth_buf[x, y] < depth:
                depth_buf[x, y] = depth
                set_pixel(x, y, color)

    else:
        for x, y in ti.ndrange((x_min, x_max + 1), (y_min, y_max + 1)):
            f23 = barycentric_ij(x, y, x2, y2, x3, y3)
            f31 = barycentric_ij(x, y, x3, y3, x1, y1)
            # f12 = barycentric_ij(x, y, x1, y1, x2, y2)

            alpha = f23 / f_alpha
            beta = f31 / f_beta
            # gamma = f12 / f_gamma
            gamma = 1 - alpha - beta

            # 在内部或边界上
            if (
                (alpha > 0 or (alpha == 0 and alpha_check))
                and (beta > 0 or (beta == 0 and beta_check))
                and (gamma > 0 or (gamma == 0 and gamma_check))
            ):
                c = alpha * c1 + beta * c2 + gamma * c3

                # 计算插值深度
                w = alpha * w1 + beta * w2 + gamma * w3
                z = alpha * z1 / w1 + beta * z2 / w2 + gamma * z3 / w3
                z /= w

                # z-test
                if depth_buf[x, y] < z:
                    depth_buf[x, y] = z
                    set_pixel(x, y, c)


# ////////// Transform /////////////
# ->> Region Start


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
def get_model_matrix():
    model = tm.mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
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


# <<- Region End
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# ///////////////////////////////////////////////
# ->> Region Start


@ti.kernel
def render():
    depth_buf.fill(-tm.inf)

    for i in indices:
        i1, i2, i3 = indices[i]
        v1, v2, v3 = (
            tm.vec4(vertices[i1], 1),
            tm.vec4(vertices[i2], 1),
            tm.vec4(vertices[i3], 1),
        )
        c1, c2, c3 = per_vertex_colors[i1], per_vertex_colors[i2], per_vertex_colors[i3]

        v1 = mvp @ v1
        v2 = mvp @ v2
        v3 = mvp @ v3

        v1 /= v1.w
        v2 /= v2.w
        v3 /= v3.w

        # set_pixel(v1.x, v1.y, c1)
        # set_pixel(v2.x, v2.y, c2)
        # set_pixel(v3.x, v3.y, c3)

        rasterize_triangle(v1, v2, v3, c1, c2, c3)


@ti.func
def set_pixel(x: int, y: int, color: ti.template()):
    if x >= 0 and x < width and y >= 0 and y < height:
        frame_buf[x, y] = color


@ti.func
def barycentric_ij(x: float, y: float, xi: float, yi: float, xj: float, yj: float):
    return (yi - yj) * x + (xj - xi) * y + xi * yj - xj * yi


# 计算三角形重心坐标
@ti.func
def compute_barycentric(
    x: float, y: float, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float
):
    alpha = barycentric_ij(x, y, x2, y2, x3, y3) / barycentric_ij(
        x1, y1, x2, y2, x3, y3
    )
    beta = barycentric_ij(x, y, x3, y3, x1, y1) / barycentric_ij(x2, y2, x3, y3, x1, y1)
    # gamma = barycentric_ij(x, y, x1, y1, x2, y2) / barycentric_ij(
    #     x3, y3, x1, y1, x2, y2
    # )
    gamma = 1 - alpha - beta
    return tm.vec3(alpha, beta, gamma)


# <<- Region End
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


if __name__ == "__main__":
    # define data
    pos = np.array(
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

    ind = np.array([(0, 1, 2), (3, 4, 5)], dtype=np.int32)

    cols = np.array(
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
    cols = cols / 255.0

    assert len(pos) == len(cols)

    # camera
    eye_pos = (0, 0, 5)
    eye_up = (0, 1, 0)
    eye_fov = 60
    eye_look_at = (0, 0, -5)
    zNear = 0.1
    zFar = 50

    # define display
    width = 2048
    height = 2048
    resolution = (width, height)
    aspect_ratio = 1.0 * height / width

    MSAA = False
    MSAA_N = 4  # MSAA-NxN

    # taichi data
    vertices = ti.Vector.field(3, float, shape=len(pos))
    indices = ti.Vector.field(3, int, shape=len(ind))
    per_vertex_colors = ti.Vector.field(3, float, shape=len(cols))

    vertices.from_numpy(pos)
    indices.from_numpy(ind)
    per_vertex_colors.from_numpy(cols)

    frame_buf = ti.Vector.field(3, float, shape=resolution)  # 屏幕像素颜色信息（rbg）
    depth_buf = ti.field(float, shape=resolution)  # 屏幕像素深度信息（z-buffer）

    # transform matrix
    viewport = get_viewport_matrix(width, height)
    view = get_view_matrix(eye_pos, eye_look_at, eye_up)
    projection = get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar)
    model = get_model_matrix()
    mvp = viewport @ projection @ view @ model

    # print("viewport:", viewport)
    # print("projection:", projection)
    # print("view:", view)
    # print("model:", model)
    # print("mvp:", mvp)

    # rendering
    window = ti.ui.Window("draw two triangles", resolution)
    canvas = window.get_canvas()

    render()

    while window.running:
        if window.is_pressed(ti.ui.ESCAPE):  # 按ESC退出
            break

        # render()
        canvas.set_image(frame_buf)
        window.show()
