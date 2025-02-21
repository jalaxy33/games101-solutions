import os

os.chdir(os.path.dirname(__file__))

import numpy as np
import taichi as ti
import taichi.math as tm
import trimesh
import skimage


ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32)

# define types
vec2 = ti.types.vector(2, float)
vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)
mat3 = ti.types.matrix(3, 3, float)
mat4 = ti.types.matrix(4, 4, float)


@ti.dataclass
class Triangle:
    # 顶点坐标
    v1: vec4
    v2: vec4
    v3: vec4
    # 顶点法向
    n1: vec4
    n2: vec4
    n3: vec4
    # 顶点颜色
    c1: vec4
    c2: vec4
    c3: vec4
    # 材质坐标
    uv1: vec2
    uv2: vec2
    uv3: vec2
    # 相机空间坐标 (viewspace position)
    vp1: vec4
    vp2: vec4
    vp3: vec4


@ti.dataclass
class Light:
    position: vec3
    intensity: vec3


@ti.dataclass
class FragmentShaderPayload:
    color: vec3
    normal: vec3
    view_pos: vec3
    tex_uv: vec2


# ////////// Shaders /////////////
# ->> Region Start


@ti.func
def get_texture_color(u: float, v: float, default_color=vec3(0)) -> vec3:
    # W, H = texture.shape[0], texture.shape[1]
    # i, j = int(u * W), int(v * H)

    # color = default_color
    # if i >= 0 and i < W and j >= 0 and j < H:
    #     color = texture[i, j]

    W, H = texture.shape[0], texture.shape[1]
    u_p, v_p = u * W - 0.5, v * H - 0.5

    iu0, iv0 = int(tm.floor(u_p)), int(tm.floor(v_p))
    iu1, iv1 = iu0 + 1, iv0 + 1

    color = default_color
    if iu0 >= 0 and iu1 < W and iv0 >= 0 and iv1 < H:
        a_u, a_v = iu1 - u_p, iv1 - v_p
        b_u, b_v = 1 - a_u, 1 - a_v
        color = (
            a_u * a_v * texture[iu0, iv0]
            + a_u * b_v * texture[iu0, iv1]
            + b_u * a_v * texture[iu1, iv0]
            + b_u * b_v * texture[iu1, iv1]
        )
    return color


# Bling-Phong Reflectance Model
@ti.func
def bling_phong_fragment_shader(payload: FragmentShaderPayload) -> vec3:
    kd = payload.color  # 调整漫反射系数

    pos = payload.view_pos
    normal = tm.normalize(payload.normal)

    pixel_color = vec3(0)
    for i in ti.ndrange(lights.shape[0]):
        # TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular*
        # components are. Then, accumulate that result on the *result_color* object.

        light = lights[i]
        light_pos = light.position.xyz
        light_intens = light.intensity.xyz

        r = tm.distance(light_pos, pos)  # 光源-点距离
        l = tm.normalize(light_pos - pos)  # 光源-点方向
        v = tm.normalize(eye_pos - pos)  # 相机-点方向
        h = tm.normalize(v + l)  # 半程向量

        E = light_intens / (r * r)  # 此处的光能

        ambient = ka * ambient_intensity
        diffuse = kd * E * tm.max(0, tm.dot(normal, l))
        specular = ks * E * tm.pow(tm.max(0, tm.dot(normal, h)), phong_exp)

        pixel_color += ambient + diffuse + specular
    return pixel_color


# 纹理映射（Texture Mapping）
@ti.func
def texture_fragment_shader(payload: FragmentShaderPayload) -> vec3:
    # TODO: Get the texture value at the texture coordinates of the current fragment
    u, v = payload.tex_uv

    default_color = payload.color
    pixel_color = get_texture_color(u, v, default_color)
    return pixel_color


# 法线贴图（Normal mapping）
@ti.func
def normal_fragment_shader(payload: FragmentShaderPayload) -> vec3:
    normal = tm.normalize(payload.normal)
    color = (normal + vec3(1)) / 2
    return color


# 凹凸贴图（Bump Mapping）
@ti.func
def bump_fragment_shader(payload: FragmentShaderPayload) -> vec3:
    # TODO: Implement bump mapping here
    # Let n = normal = (x, y, z)
    # Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    # Vector b = n cross product t
    # Matrix TBN = [t b n]
    # dU = kh * kn * (h(u+1/w,v)-h(u,v))
    # dV = kh * kn * (h(u,v+1/h)-h(u,v))
    # Vector ln = (-dU, -dV, 1)
    # Normal n = normalize(TBN * ln)

    W, H = texture.shape[0], texture.shape[1]
    u, v = payload.tex_uv

    pos = payload.view_pos.xyz
    color = payload.color
    n = tm.normalize(payload.normal.xyz)

    # 计算新法向量
    sqrt_xz = tm.sqrt(n.x * n.x + n.z * n.z)
    t = vec3(n.x * n.y / sqrt_xz, sqrt_xz, n.z * n.y / sqrt_xz)
    b = tm.cross(n, t)

    TBN = mat3(
        [
            [t.x, b.x, n.x],
            [t.y, b.y, n.y],
            [t.z, b.z, n.z],
        ]
    )

    default_color = color

    dU = (
        kh
        * kn
        * (
            get_texture_color(tm.min(u + 1.0 / W, 1.0), v, default_color).norm()
            - get_texture_color(u, v, default_color).norm()
        )
    )
    dV = (
        kh
        * kn
        * (
            get_texture_color(u, tm.min(v + 1.0 / H, 1.0), default_color).norm()
            - get_texture_color(u, v, default_color).norm()
        )
    )

    ln = vec3(-dU, -dV, 1)
    new_norm = tm.normalize(TBN @ ln)
    pixel_color = new_norm
    return pixel_color


# 位移贴图（Displacement Mapping）
@ti.func
def displacement_fragment_shader(payload: FragmentShaderPayload) -> vec3:
    # TODO: Implement bump mapping here
    # Let n = normal = (x, y, z)
    # Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    # Vector b = n cross product t
    # Matrix TBN = [t b n]
    # dU = kh * kn * (h(u+1/w,v)-h(u,v))
    # dV = kh * kn * (h(u,v+1/h)-h(u,v))
    # Vector ln = (-dU, -dV, 1)
    # Normal n = normalize(TBN * ln)
    W, H = texture.shape[0], texture.shape[1]
    u, v = payload.tex_uv

    pos = payload.view_pos.xyz
    color = payload.color
    n = tm.normalize(payload.normal.xyz)

    # 计算新法向量
    sqrt_xz = tm.sqrt(n.x * n.x + n.z * n.z)
    t = vec3(n.x * n.y / sqrt_xz, sqrt_xz, n.z * n.y / sqrt_xz)
    b = tm.cross(n, t)

    TBN = mat3(
        [
            [t.x, b.x, n.x],
            [t.y, b.y, n.y],
            [t.z, b.z, n.z],
        ]
    )

    default_color = color

    dU = (
        kh
        * kn
        * (
            get_texture_color(tm.min(u + 1.0 / W, 1.0), v, default_color).norm()
            - get_texture_color(u, v, default_color).norm()
        )
    )
    dV = (
        kh
        * kn
        * (
            get_texture_color(u, tm.min(v + 1.0 / H, 1.0), default_color).norm()
            - get_texture_color(u, v, default_color).norm()
        )
    )

    ln = vec3(-dU, -dV, 1)
    new_norm = tm.normalize(TBN @ ln)

    pixel_color = color
    for i in ti.ndrange(lights.shape[0]):
        # TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular*
        # components are. Then, accumulate that result on the *result_color* object.

        light = lights[i]
        light_pos = light.position.xyz
        light_intens = light.intensity.xyz

        r = tm.distance(light_pos, pos)  # 光源-点距离
        l = tm.normalize(light_pos - pos)  # 光源-点方向
        v = tm.normalize(eye_pos - pos)  # 相机-点方向
        h = tm.normalize(v + l)  # 半程向量

        E = light_intens / (r * r)  # 此处的光能

        ambient = ka * ambient_intensity
        diffuse = kd * E * tm.max(0, tm.dot(new_norm, l))
        specular = ks * E * tm.pow(tm.max(0, tm.dot(new_norm, h)), phong_exp)

        pixel_color += ambient + diffuse + specular
    return pixel_color


# <<- Region End
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# ////////// Transform /////////////
# ->> Region Start


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
    w = -normalize(look_at)
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
    angles=vec3(0, 0, 0),
    scales=vec3(1, 1, 1),
    translates=vec3(0, 0, 0),
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
def get_orthographic_matrix(
    eye_fov: float, aspect_ratio: float, zNear: float, zFar: float
) -> mat4:
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
def get_projection_matrix(
    eye_fov: float, aspect_ratio: float, zNear: float, zFar: float
) -> mat4:
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


# <<- Region End
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# ///////////////////////////////////////////////
# ->> Region Start


@ti.func
def set_pixel(x: int, y: int, color: ti.template()):
    if x >= 0 and x < width and y >= 0 and y < height:
        frame_buf[x, y] = color


@ti.func
def barycentric_ij(x: float, y: float, xi: float, yi: float, xj: float, yj: float):
    return (yi - yj) * x + (xj - xi) * y + xi * yj - xj * yi


@ti.func
def rasterize_triangle(t: Triangle):
    # TODO: From your HW2, get the triangle rasterization code.
    # TODO: Inside your rasterization loop:
    #    * v[i].w() is the vertex view space depth value z.
    #    * Z is interpolated view space depth for the current pixel
    #    * zp is depth between zNear and zFar, used for z-buffer

    # float Z = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    # float zp = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    # zp *= Z;

    # TODO: Interpolate the attributes:
    # auto interpolated_color
    # auto interpolated_normal
    # auto interpolated_texcoords
    # auto interpolated_shadingcoords

    # Use: fragment_shader_payload payload( interpolated_color, interpolated_normal.normalized(), interpolated_texcoords, texture ? &*texture : nullptr);
    # Use: payload.view_pos = interpolated_shadingcoords;
    # Use: Instead of passing the triangle's color directly to the frame buffer, pass the color to the shaders first to get the final color;
    # Use: auto pixel_color = fragment_shader(payload);

    v1, v2, v3 = t.v1, t.v2, t.v3
    n1, n2, n3 = t.n1, t.n2, t.n3
    c1, c2, c3 = t.c1, t.c2, t.c3
    uv1, uv2, uv3 = t.uv1, t.uv2, t.uv3
    vp1, vp2, vp3 = t.vp1, t.vp2, t.vp3

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

            depth = -tm.inf  # store depth
            pixel_color = vec3(0)  # pixel color
            for i, j in ti.ndrange(MSAA_N, MSAA_N):
                x_curr = x_start + i * step_size
                y_curr = y_start + j * step_size

                f23 = barycentric_ij(x_curr, y_curr, x2, y2, x3, y3)
                f31 = barycentric_ij(x_curr, y_curr, x3, y3, x1, y1)

                alpha = f23 / f_alpha
                beta = f31 / f_beta
                gamma = 1 - alpha - beta

                # 在内部或边界上
                if (
                    (alpha > 0 or (alpha == 0 and alpha_check))
                    and (beta > 0 or (beta == 0 and beta_check))
                    and (gamma > 0 or (gamma == 0 and gamma_check))
                ):
                    # 计算插值
                    c = alpha * c1 + beta * c2 + gamma * c3  # 颜色
                    n = alpha * n1 + beta * n2 + gamma * n3  # 法向
                    uv = alpha * uv1 + beta * uv2 + gamma * uv3  # 材质坐标
                    vp = alpha * vp1 + beta * vp2 + gamma * vp3  # 相机空间坐标

                    # 计算插值深度
                    w = alpha * w1 + beta * w2 + gamma * w3
                    z = alpha * z1 / w1 + beta * z2 / w2 + gamma * z3 / w3
                    z /= w

                    if depth < z:
                        depth = z

                        payload = FragmentShaderPayload(
                            color=c.xyz, normal=n.xyz, view_pos=vp.xyz, tex_uv=uv
                        )

                        curr_color = c.xyz
                        if shader_option[None] == 1:  # Bling-phong
                            curr_color = bling_phong_fragment_shader(payload)
                        elif shader_option[None] == 2:  # 纹理映射
                            curr_color = texture_fragment_shader(payload)
                        elif shader_option[None] == 3:  # 法线映射
                            curr_color = normal_fragment_shader(payload)
                        elif shader_option[None] == 4:  # 凹凸贴图
                            curr_color = bump_fragment_shader(payload)
                        elif shader_option[None] == 5:  # 位移贴图
                            curr_color = displacement_fragment_shader(payload)
                        pixel_color += curr_color

            pixel_color /= MSAA_N * MSAA_N  # 周围超像素的颜色取平均
            if depth_buf[x, y] <= depth:
                depth_buf[x, y] = depth
                set_pixel(x, y, pixel_color)

    else:
        for x, y in ti.ndrange((x_min, x_max + 1), (y_min, y_max + 1)):
            f23 = barycentric_ij(x, y, x2, y2, x3, y3)
            f31 = barycentric_ij(x, y, x3, y3, x1, y1)

            alpha = f23 / f_alpha
            beta = f31 / f_beta
            gamma = 1 - alpha - beta

            # 在内部或边界上
            if (
                (alpha > 0 or (alpha == 0 and alpha_check))
                and (beta > 0 or (beta == 0 and beta_check))
                and (gamma > 0 or (gamma == 0 and gamma_check))
            ):
                # 计算插值
                c = alpha * c1 + beta * c2 + gamma * c3  # 颜色
                n = alpha * n1 + beta * n2 + gamma * n3  # 法向
                uv = alpha * uv1 + beta * uv2 + gamma * uv3  # 材质坐标
                vp = alpha * vp1 + beta * vp2 + gamma * vp3  # 相机空间坐标

                # 计算插值深度
                w = alpha * w1 + beta * w2 + gamma * w3
                z = alpha * z1 / w1 + beta * z2 / w2 + gamma * z3 / w3
                z /= w

                # z-test
                if depth_buf[x, y] < z:
                    depth_buf[x, y] = z

                    payload = FragmentShaderPayload(
                        color=c.xyz, normal=n.xyz, view_pos=vp.xyz, tex_uv=uv
                    )

                    pixel_color = c.xyz
                    if shader_option[None] == 1:  # Bling-phong
                        pixel_color = bling_phong_fragment_shader(payload)
                    elif shader_option[None] == 2:  # 纹理映射
                        pixel_color = texture_fragment_shader(payload)
                    elif shader_option[None] == 3:  # 法线贴图
                        pixel_color = normal_fragment_shader(payload)
                    elif shader_option[None] == 4:  # 凹凸贴图
                        pixel_color = bump_fragment_shader(payload)
                    elif shader_option[None] == 5:  # 位移贴图
                        pixel_color = displacement_fragment_shader(payload)

                    set_pixel(x, y, pixel_color)


@ti.kernel
def render():
    frame_buf.fill(background_color)
    depth_buf.fill(-tm.inf)

    view_model = view @ model
    inv_trans = view_model.inverse().transpose()

    # for i in ti.ndrange(1):
    for i in indices:
        t = Triangle()

        i1, i2, i3 = indices[i]
        v1, v2, v3 = (
            vec4(vertices[i1], 1),
            vec4(vertices[i2], 1),
            vec4(vertices[i3], 1),
        )
        n1, n2, n3 = (
            inv_trans @ vec4(normals[i1], 0),
            inv_trans @ vec4(normals[i2], 0),
            inv_trans @ vec4(normals[i3], 0),
        )
        vp1, vp2, vp3 = (
            view_model @ v1,
            view_model @ v2,
            view_model @ v3,
        )
        c1, c2, c3 = per_vertex_colors[i1], per_vertex_colors[i2], per_vertex_colors[i3]
        uv1, uv2, uv3 = textUVs[i1], textUVs[i2], textUVs[i3]

        v1 = mvp @ v1
        v2 = mvp @ v2
        v3 = mvp @ v3

        v1 /= v1.w
        v2 /= v2.w
        v3 /= v3.w

        t.v1, t.v2, t.v3 = v1, v2, v3
        t.n1, t.n2, t.n3 = n1, n2, n3
        t.c1, t.c2, t.c3 = c1, c2, c3
        t.uv1, t.uv2, t.uv3 = uv1, uv2, uv3
        t.vp1, t.vp2, t.vp3 = vp1, vp2, vp3

        rasterize_triangle(t)


# <<- Region End
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


if __name__ == "__main__":
    # load model
    obj_file = os.path.abspath("../task/Code/models/spot/spot_triangulated_good.obj")
    texture_png = os.path.abspath("../task/Code/models/spot/spot_texture.png")
    mesh = trimesh.load_mesh(obj_file)
    tex_img = skimage.io.imread(texture_png).astype(np.float32) / 255  # H, W, C

    verts = mesh.vertices.astype(np.float32)
    inds = mesh.faces.astype(np.int32)
    norms = mesh.vertex_normals.astype(np.float32)
    uvs = mesh.visual.uv.astype(np.float32)
    vert_colors = mesh.visual.to_color().vertex_colors.astype(np.float32) / 255

    tex_img = tex_img.transpose(1, 0, 2)[:, ::-1, :]

    # import matplotlib.pyplot as plt
    # plt.imshow(tex_img)
    # plt.show()

    # print(verts.shape, inds.shape, norms.shape, uvs.shape)
    # mesh.show()

    # camera
    eye_pos = (0, 0, 10)
    eye_up = (0, 1, 0)
    eye_fov = 45
    eye_look_at = (0, 0, -5)
    zNear = 0.1
    zFar = 50

    # lighting
    lights = Light.field(shape=2)
    lights[0] = Light(position=vec3(20, 20, 20), intensity=(800, 800, 800))
    lights[1] = Light(position=vec3(-20, 20, 0), intensity=(800, 800, 800))

    kd = vec3(1)  # 漫反射系数
    ks = vec3(0.7937, 0.7937, 0.7937)  # 镜面反射系数
    ka = vec3(0.005, 0.005, 0.005)  # 环境光系数
    ambient_intensity = vec3(10, 10, 10)  # 环境光强度
    phong_exp = 150.0  # 高光区域集中度

    # bump mapping & displacement mapping
    kh = 0.2
    kn = 0.1

    # define display
    width = 1024
    height = 1024
    resolution = (width, height)
    aspect_ratio = 1.0 * height / width

    background_color = vec3(0)

    MSAA = False
    MSAA_N = 2  # MSAA-NxN

    # data transform
    angles = (0, 140, 0)
    scales = vec3(2.5, 2.5, 2.5)
    translates = vec3(0, 0, 0)

    # taichi data
    vertices = ti.Vector.field(3, float, shape=len(verts))
    indices = ti.Vector.field(3, int, shape=len(inds))
    normals = ti.Vector.field(3, float, shape=len(norms))
    textUVs = ti.Vector.field(2, float, shape=len(uvs))
    per_vertex_colors = ti.Vector.field(4, float, shape=len(vert_colors))
    texture = ti.Vector.field(3, float, shape=(tex_img.shape[0], tex_img.shape[1]))

    vertices.from_numpy(verts)
    indices.from_numpy(inds)
    normals.from_numpy(norms)
    textUVs.from_numpy(uvs)
    per_vertex_colors.from_numpy(vert_colors)
    texture.from_numpy(tex_img)

    frame_buf = ti.Vector.field(3, float, shape=resolution)  # 屏幕像素颜色信息（rbg）
    depth_buf = ti.field(float, shape=resolution)  # 屏幕像素深度信息（z-buffer）

    # 着色器选项（0-默认, 1-Bling-Phong，2-纹理映射, 3-法线贴图，4-凹凸贴图，5-位移贴图）
    shader_option = ti.field(int, shape=())
    shader_option[None] = 0

    # transform matrix
    viewport = get_viewport_matrix(width, height)
    view = get_view_matrix(eye_pos, eye_look_at, eye_up)
    projection = get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar)
    model = get_model_matrix(angles, scales, translates)
    mvp = viewport @ projection @ view @ model

    # print("viewport:", viewport)
    # print("projection:", projection)
    # print("view:", view)
    # print("model:", model)
    # print("mvp:", mvp)

    for i in ti.ndrange(lights.shape[0]):
        lights[i].position = view @ lights[i].position

    # rendering
    window = ti.ui.Window("Render a cow", resolution)
    canvas = window.get_canvas()

    render()

    while window.running:
        if window.is_pressed(ti.ui.ESCAPE):  # 按ESC退出
            break

        # 选择着色器
        switch_shader = False
        if window.is_pressed(ti.ui.SPACE):  # 按空格复原
            shader_option[None] = 0
            switch_shader = True
        if window.is_pressed("q"):
            shader_option[None] = 1
            switch_shader = True
        if window.is_pressed("w"):
            shader_option[None] = 2
            switch_shader = True
        if window.is_pressed("e"):
            shader_option[None] = 3
            switch_shader = True
        if window.is_pressed("r"):
            shader_option[None] = 4
            switch_shader = True
        if window.is_pressed("t"):
            shader_option[None] = 5
            switch_shader = True

        if switch_shader:
            render()

        canvas.set_image(frame_buf)
        window.show()
