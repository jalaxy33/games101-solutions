# Assignment 5. Whitted-style RayTracer
# Reference:
#   https://raytracing.github.io/books/RayTracingInOneWeekend.html


import taichi as ti
import taichi.math as tm
from collections import deque

from models import *
from renderer import *

ti.init(arch=ti.gpu)


# Variables
WIDTH = 1280
HEIGHT = 960
background_color = (0.235294, 0.67451, 0.843137)
MAX_DEPTH = 10
MAX_STACK_SIZE = 50
EPS = 1e-9
SAMPLES_PER_PIXEL = 5


# Global Variables
frame_buf = ti.Vector.field(3, ti.f32, shape=(WIDTH, HEIGHT))


@ti.func
def rand3():
    return ti.Vector([ti.random(), ti.random(), ti.random()])


@ti.func
def random_in_unit_sphere():
    p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    while p.norm() >= 1.0:
        p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    return p


@ti.func
def random_unit_vector():
    return random_in_unit_sphere().normalized()


@ti.dataclass
class AmbientLight:
    Ia: tm.vec3


@ti.dataclass
class PointLight:
    pos: tm.vec3  # position
    I: tm.vec3  # Intensity


@ti.data_oriented
class LightList:
    def __init__(self):
        self.ambient_light = AmbientLight()
        self.point_lights = []

    def add_ambient_light(self, intensity: tm.vec3):
        self.ambient_light.Ia = tm.vec3(intensity)

    def add_point_light(self, pos: tm.vec3, intensity: tm.vec3):
        self.point_lights.append(PointLight(tm.vec3(pos), tm.vec3(intensity)))


lights = LightList()


@ti.data_oriented
class HittableList:
    def __init__(self):
        self.objs = []

    def clear(self):
        self.objs.clear()

    def add(self, obj):
        self.objs.append(obj)

    @ti.func
    def hit(self, ray: Ray, tmin=tm.inf) -> HitRecord:
        rec = HitRecord()
        closest_so_far = tmin

        for i in ti.static(range(len(self.objs))):
            temp_rec = self.objs[i].hit(ray, closest_so_far)
            if temp_rec.is_hit and temp_rec.t < closest_so_far:
                closest_so_far = temp_rec.t
                rec = temp_rec

        return rec


world = HittableList()


@ti.func
def reflect(v, n):
    return v - 2 * v.dot(n) * n


@ti.func
def refract(v, n, etai_over_etat):
    cos_theta = min(n.dot(-v), 1.0)
    r_out_perp = etai_over_etat * (v + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel


@ti.func
def reflectance(cosine, ref_idx):
    # Use Schlick's approximation for reflectance.
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)


@ti.func
def bling_phong(ray: Ray, rec: HitRecord):
    eps = 1e-5
    shadow_orig = (
        rec.pos + rec.N * eps if tm.dot(ray.rd, rec.N) < 0 else rec.pos - rec.N * eps
    )

    color = tm.vec3(0)
    ambient_color = tm.vec3(0)
    diffuse_color = tm.vec3(0)
    specular_color = tm.vec3(0)
    for i in ti.static(range(len(lights.point_lights))):
        light = lights.point_lights[i]

        r = tm.distance(light.pos, rec.pos)  # 光源-点距离
        light_dir = (light.pos - rec.pos).normalized()  # 光源-点方向

        shadow_rec = world.hit(Ray(shadow_orig, light_dir), tm.inf)
        in_shadow = shadow_rec.is_hit and shadow_rec.t < r

        # ambient
        ambient_color = rec.mat.ka * lights.ambient_light.Ia

        # diffuse
        diffuse_color += tm.vec3(0) if in_shadow else max(0, tm.dot(light_dir, rec.N))

        # specular
        reflect_dir = reflect(-light_dir, rec.N).normalized()
        specular_color += light.I * tm.pow(
            tm.max(0, -tm.dot(reflect_dir, light_dir)), rec.mat.spec_exp
        )

    color = (
        ambient_color
        + rec.mat.kd * rec.mat.diffuse_color * diffuse_color
        + rec.mat.ks * specular_color
    )

    return color


# @ti.dataclass
# class Stack:
#     def __init__(self, value_type, width, height, max_depth):
#         self.max_depth = max_depth
#         self.stack = ti.field(value_type, shape=(width, height, max_depth))
#         self.curr_depth = ti.field(int, shape=(width, height))

#         self.curr_depth.fill(-1)

#     @ti.func
#     def clear(self, i, j):
#         self.curr_depth[i, j] = -1

#     @ti.func
#     def push(self, i, j, value) -> bool:
#         succeed = False
#         if self.curr_depth[i, j] + 1 < self.max_depth:
#             succeed = True
#             self.curr_depth[i, j] += 1
#             self.stack[i, j, self.curr_depth[i, j]] = value
#         return succeed

#     @ti.func
#     def pop(self, i, j) -> bool:
#         succeed = False
#         if self.curr_depth[i, j] - 1 >= -1:
#             self.curr_depth[i, j] -= 1
#         return succeed

#     @ti.func
#     def top(self, i, j):
#         return self.stack[i, j, self.curr_depth[i, j]]

#     @ti.func
#     def depth(self, i, j) -> int:
#         return self.curr_depth[i, j]

#     @ti.func
#     def empty(self, i, j) -> bool:
#         return self.curr_depth[i, j] <= -1

#     @ti.func
#     def full(self, i, j) -> bool:
#         return self.curr_depth[i, j] >= self.max_depth


@ti.dataclass
class StatusFrame:
    phase: ti.i32
    depth: ti.i32
    ray: Ray
    weight: ti.f32


Color = ti.types.vector(3, ti.f32)


task_stack = StatusFrame.field(shape=(WIDTH, HEIGHT, MAX_STACK_SIZE))
task_stack_ptr = ti.field(ti.i32, shape=(WIDTH, HEIGHT))
task_stack_ptr.fill(-1)

result_stack = ti.field(Color, shape=(WIDTH, HEIGHT, MAX_STACK_SIZE))
result_stack_ptr = ti.field(ti.i32, shape=(WIDTH, HEIGHT))
result_stack_ptr.fill(-1)


@ti.func
def task_stack_clear(i, j):
    task_stack_ptr[i, j] = -1


@ti.func
def result_stack_clear(i, j):
    result_stack_ptr[i, j] = -1


@ti.func
def task_stack_push(i, j, value) -> bool:
    succeed = False
    if task_stack_ptr[i, j] + 1 < MAX_STACK_SIZE:
        succeed = True
        task_stack_ptr[i, j] += 1
        task_stack[i, j, task_stack_ptr[i, j]] = value
    return succeed


@ti.func
def result_stack_push(i, j, value) -> bool:
    succeed = False
    if result_stack_ptr[i, j] + 1 < MAX_STACK_SIZE:
        succeed = True
        result_stack_ptr[i, j] += 1
        result_stack[i, j, result_stack_ptr[i, j]] = Color(value)
    return succeed


@ti.func
def task_stack_pop(i, j) -> bool:
    succeed = False
    if task_stack_ptr[i, j] - 1 >= -1:
        succeed = True
        task_stack_ptr[i, j] -= 1
    return succeed


@ti.func
def task_stack_pop(i, j) -> bool:
    succeed = False
    if result_stack_ptr[i, j] - 1 >= -1:
        succeed = True
        result_stack_ptr[i, j] -= 1
    return succeed


@ti.func
def task_stack_top(i, j) -> StatusFrame:
    return task_stack[i, j, task_stack_ptr[i, j]]


@ti.func
def result_stack_top(i, j):
    return result_stack[i, j, result_stack_ptr[i, j]]


@ti.func
def task_stack_empty(i, j):
    return task_stack_ptr[i, j] <= -1


@ti.func
def result_stack_empty(i, j) -> bool:
    return result_stack_ptr[i, j] <= -1


# # Whitted-style Raytracer
# @ti.func
# def ray_color(i, j, ray: Ray):
#     eps = 1e-5
#     PROCESS_PHASE = 0
#     COMBINE_PHASE = 1

#     task_stack.clear(i, j)
#     result_stack.clear(i, j)
#     pixel_color = tm.vec3(0)

#     init_frame = StackFrame(phase=PROCESS_PHASE, depth=0, ray=ray, weight=1.0)
#     task_stack.push(i, j, init_frame)
#     while not task_stack.empty(i, j):
#         curr_frame = task_stack.top(i, j)
#         task_stack.pop(i, j)

#         if curr_frame.depth >= MAX_DEPTH:
#             continue

#         rec = world.hit(curr_frame.ray, tm.inf)
#         rd = curr_frame.ray.rd.normalized()

#         if rec.is_hit:
#             # Phase 0: Process Raytracing
#             if curr_frame.phase == PROCESS_PHASE:
#                 if rec.mat.m_type == REFLECTION_AND_REFRACTION:
#                     ref_idx = rec.mat.ior
#                     if rec.front_face:
#                         ref_idx = 1.0 / ref_idx

#                     cos_theta = max(min(-rd.dot(rec.N), 1.0), -1.0)

#                     kr = reflectance(cos_theta, ref_idx)

#                     reflect_dir = reflect(rd, rec.N).normalized()
#                     reflect_orig = rec.pos
#                     if reflect_dir.dot(rec.N) > 0:
#                         reflect_orig += rec.N * eps
#                     else:
#                         reflect_orig -= rec.N * eps

#                     refract_dir = refract(rd, rec.N, ref_idx).normalized()
#                     refract_orig = rec.pos
#                     if refract_dir.dot(rec.N) > 0:
#                         refract_orig += rec.N * eps
#                     else:
#                         refract_orig -= rec.N * eps

#                     reflect_frame = StackFrame(
#                         phase=PROCESS_PHASE,
#                         depth=curr_frame.depth + 1,
#                         ray=Ray(reflect_orig, reflect_dir),
#                     )

#                     refract_frame = StackFrame(
#                         phase=PROCESS_PHASE,
#                         depth=curr_frame.depth + 1,
#                         ray=Ray(refract_orig, refract_dir),
#                     )

#                     # Push Combine task first, then refraction and reflection
#                     task_stack.push(
#                         i,
#                         j,
#                         StackFrame(
#                             phase=COMBINE_PHASE, depth=curr_frame.depth + 1, kr=kr
#                         ),
#                     )
#                     task_stack.push(i, j, reflect_frame)
#                     task_stack.push(i, j, refract_frame)
#                 else:
#                     local_color = bling_phong(ray, rec)
#                     result_stack.push(i, j, local_color)

#             # Phase 1: Combine
#             elif curr_frame.phase == COMBINE_PHASE:
#                 kr = curr_frame.kr
#                 refract_color = result_stack.top(i, j)
#                 result_stack.pop(i, j)

#                 reflect_color = result_stack.top(i, j)
#                 result_stack.pop(i, j)

#                 combined_color = reflect_color * kr + refract_color * (1 - kr)
#                 result_stack.push(i, j, combined_color)

#         else:
#             break

#     if not result_stack.empty(i, j):
#         pixel_color = result_stack.top(i, j)
#     else:
#         pixel_color = background_color
#     return pixel_color


# Whitted-style Raytracer
@ti.func
def ray_color(i, j, ray: Ray):
    eps = 1e-5
    PROCESS_PHASE = 0
    MERGE_PHASE = 1

    task_stack_clear(i, j)
    result_stack_clear(i, j)
    pixel_color = tm.vec3(0)

    init_frame = StatusFrame(phase=PROCESS_PHASE, depth=0, ray=ray, weight=1.0)
    task_stack_push(i, j, init_frame)
    while not task_stack_empty(i, j):
        #     curr_frame = task_stack_top(i, j)
        task_stack_pop(i, j)

    #     if curr_frame.phase == PROCESS_PHASE:
    #         rec = world.hit(curr_frame.ray, tm.inf)
    #         rd = curr_frame.ray.rd.normalized()
    #         m_type = rec.mat.m_type

    #         if curr_frame.depth + 1 >= MAX_DEPTH:
    #             result_stack_push(i, j, Color(0, 0, 0))
    #         elif m_type == REFLECTION_AND_REFRACTION:
    #             pass
    #         elif m_type == REFLECTION:
    #             pass
    #         else:
    #             pass

    #     elif curr_frame.phase == MERGE_PHASE:
    #         pass

    return pixel_color


# # Whitted-style Raytracer
# @ti.func
# def ray_color(i, j, ray: Ray):
#     pixel_color = tm.vec3(0)
#     accum_factor = 1.0

#     curr_ray = Ray(ray.ro, ray.rd.normalized())

#     for depth in range(MAX_DEPTH):
#         rec = world.hit(curr_ray, tm.inf)
#         if rec.is_hit:
#             if rec.mat.m_type == REFLECTION_AND_REFRACTION:
#                 ref_idx = rec.mat.ior
#                 if rec.front_face:
#                     ref_idx = 1.0 / ref_idx
#                 rd = curr_ray.rd.normalized()
#                 cos_theta = min(-rd.dot(rec.N), 1.0)
#                 sin_theta = tm.sqrt(1 - cos_theta * cos_theta)
#                 kr = reflectance(cos_theta, ref_idx)
#                 if ref_idx * sin_theta > 1.0 or reflectance(cos_theta, ref_idx) > ti.random():
#                     reflect_dir = reflect(rd, rec.N).normalized()
#                     reflect_orig = rec.pos
#                     curr_ray = Ray(reflect_orig, reflect_dir)
#                     accum_factor *= kr
#                 else:
#                     refract_dir = refract(rd, rec.N, ref_idx).normalized()
#                     refract_orig = rec.pos
#                     curr_ray = Ray(refract_orig, refract_dir)
#                     accum_factor *= 1 - kr
#             else:
#                 pixel_color = bling_phong(curr_ray, rec) * accum_factor
#                 break
#         else:
#             pixel_color = background_color
#             break

#     return pixel_color


@ti.kernel
def render():
    vfov = 90
    eye_pos = tm.vec3(0, 0, 0)

    aspect_ratio = 1.0 * WIDTH / HEIGHT
    half_height = tm.tan(tm.radians(vfov / 2))
    half_width = half_height * aspect_ratio

    for i, j in frame_buf:
        # TODO: Find the x and y positions of the current pixel to get the direction
        # vector that passes through it.
        # Also, don't forget to multiply both of them with the variable *scale*, and
        # x (horizontal) variable with the *imageAspectRatio*
        pixel_color = tm.vec3(0)

        for _ in range(SAMPLES_PER_PIXEL):
            x = ((i + ti.random()) / WIDTH - 0.5) * 2 * half_width
            y = ((j + ti.random()) / HEIGHT - 0.5) * 2 * half_height

            # ray_direction = cam_lower_left_corner + x * cam_horizontal + y * cam_vertical
            ray_direction = tm.vec3(x, y, -1).normalized()
            ray = Ray(eye_pos, ray_direction)

            pixel_color += ray_color(i, j, ray)

        pixel_color /= SAMPLES_PER_PIXEL
        frame_buf[i, j] = pixel_color


def init_scene():
    sph1 = Sphere((-1, 0, -12), 2)
    sph1.set_material(m_type=DIFFUSE_AND_GLOSSY, diffuse_color=(0.6, 0.7, 0.8))

    sph2 = Sphere((0.5, -0.5, -8), 1.5)
    sph2.set_material(m_type=REFLECTION_AND_REFRACTION, ior=1.5)

    mesh = MeshTriangle(
        vertices=[[-5, -3, -6], [5, -3, -6], [5, -3, -16], [-5, -3, -16]],
        indices=[[0, 1, 3], [1, 2, 3]],
        st_coords=[[0, 0], [1, 0], [1, 1], [0, 1]],
    )
    mesh.set_material(m_type=DIFFUSE_AND_GLOSSY)

    world.add(sph1)
    world.add(sph2)
    world.add(mesh)

    lights.add_ambient_light(10)
    lights.add_point_light((-20, 70, 20), 0.5)
    lights.add_point_light((30, 50, -12), 0.5)


if __name__ == "__main__":

    init_scene()

    window = ti.ui.Window("Whitted-Style Raytracer", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    render()

    while window.running:
        if window.is_pressed(ti.ui.ESCAPE):
            break

        canvas.set_image(frame_buf)
        window.show()
