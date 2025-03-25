import taichi as ti
import taichi.math as tm
import numpy as np

from Scene import Scene
from Material import *

Inf = float("inf")
Vec3 = ti.types.vector(3, dtype=ti.f32)
Color = ti.types.vector(3, float)


# Define task types
TASK_PROCESS = 0  # Process a material
TASK_MERGE = 1  # Merge results from REFLECT_AND_REFRACT


@ti.pyfunc
def clamp(x, x_min, x_max):
    return max(x_min, min(x_max, x))


@ti.pyfunc
def reflectance(cosine, ref_idx):
    # Use Schlick's approximation for reflectance.
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)


@ti.dataclass
class StatusFrame:
    task_type: ti.i32
    depth: ti.i32
    ray_origin: Vec3
    ray_direct: Vec3
    kr: ti.f32


@ti.data_oriented
class StatusStack:
    def __init__(self, width, height, max_depth):
        self.max_depth = max_depth
        self.stack = StatusFrame.field(shape=(width, height, max_depth))
        self.ptr = ti.field(ti.i32, shape=(width, height))

        self.ptr.fill(-1)

    @ti.func
    def clear(self, i, j):
        self.ptr[i, j] = -1

    @ti.func
    def empty(self, i, j) -> bool:
        return self.ptr[i, j] <= -1

    @ti.func
    def full(self, i, j) -> bool:
        return self.ptr[i, j] >= self.max_depth

    @ti.func
    def top(self, i, j):
        return self.stack[i, j, self.ptr[i, j]]

    @ti.func
    def push(self, i, j, value) -> bool:
        succeed = False
        if self.ptr[i, j] + 1 < self.max_depth:
            succeed = True
            self.ptr[i, j] += 1
            self.stack[i, j, self.ptr[i, j]] = value
        return succeed

    @ti.func
    def pop(self, i, j) -> bool:
        succeed = False
        if self.ptr[i, j] - 1 >= -1:
            succeed = True
            self.ptr[i, j] -= 1
        return succeed


@ti.data_oriented
class ResultStack:
    def __init__(self, width, height, max_depth):
        self.max_depth = max_depth
        self.stack = ti.field(Color, shape=(width, height, max_depth))
        self.ptr = ti.field(ti.i32, shape=(width, height))

        self.ptr.fill(-1)

    @ti.func
    def clear(self, i, j):
        self.ptr[i, j] = -1

    @ti.func
    def empty(self, i, j) -> bool:
        return self.ptr[i, j] <= -1

    @ti.func
    def full(self, i, j) -> bool:
        return self.ptr[i, j] >= self.max_depth

    @ti.func
    def top(self, i, j):
        return self.stack[i, j, self.ptr[i, j]]

    @ti.func
    def push(self, i, j, value) -> bool:
        succeed = False
        if self.ptr[i, j] + 1 < self.max_depth:
            succeed = True
            self.ptr[i, j] += 1
            self.stack[i, j, self.ptr[i, j]] = value
        return succeed

    @ti.func
    def pop(self, i, j) -> bool:
        succeed = False
        if self.ptr[i, j] - 1 >= -1:
            succeed = True
            self.ptr[i, j] -= 1
        return succeed


@ti.data_oriented
class Renderer:
    def __init__(
        self,
        width,
        height,
        background_color=(0.235294, 0.67451, 0.843137),
        max_depth=10,
        max_stack_size=50,
        vfov=90,
        eye_pos=(0, 0, 0),
    ):
        self.W = width
        self.H = height
        self.background_color = Vec3(background_color)
        self.max_depth = max_depth
        self.max_stack_size = max_stack_size
        self.vfov = vfov
        self.eye_pos = Vec3(eye_pos)

        self.frame_buf = ti.Vector.field(3, ti.f32, shape=(width, height))
        self.status_stack = StatusStack(width, height, max_stack_size)
        self.result_stack = ResultStack(width, height, max_stack_size)

    def set_scene(self, scene: Scene):
        self.lights = scene.lights
        self.world = scene.world

    @ti.func
    def bling_phong(self, ray_direction, hit_pos, hit_norm, hit_mat: Material) -> Vec3:
        ray_direction = ray_direction.normalized()
        hit_norm = hit_norm.normalized()

        eps = 1e-5
        shadow_origin = hit_pos + hit_norm * eps if ray_direction.dot(hit_norm) < 0 else hit_pos - hit_norm * eps

        ambient = Vec3(0)
        diffuse = Vec3(0)
        specular = Vec3(0)
        for i in ti.static(range(len(self.lights.point_lights))):
            light = self.lights.point_lights[i]

            r = tm.distance(light.pos, hit_pos)  # 光源-点距离
            light_dir = (light.pos - hit_pos).normalized()  # 光源-点方向

            in_shadow, _t, _pos, _norm, _mat = self.world.hit(shadow_origin, light_dir, t_near=r)

            # ambient
            ambient += hit_mat.ka * self.lights.ambient_light.Ia

            # diffuse
            diffuse += Vec3(0) if in_shadow else Vec3(max(0, light_dir.dot(hit_norm)))

            # specular
            reflect_dir = tm.reflect(-light_dir, hit_norm).normalized()
            specular += light.I * tm.pow(max(0, -reflect_dir.dot(light_dir)), hit_mat.spec_exp)

        diffuse *= hit_mat.kd * hit_mat.m_color
        specular *= hit_mat.ks
        color = ambient + diffuse + specular
        return color

    @ti.kernel
    def render(self):
        aspect_ratio = 1.0 * self.W / self.H
        half_height = tm.tan(tm.radians(self.vfov / 2))
        half_width = half_height * aspect_ratio

        for i, j in self.frame_buf:
            # TODO: Find the x and y positions of the current pixel to get the direction
            # vector that passes through it.
            # Also, don't forget to multiply both of them with the variable *scale*, and
            # x (horizontal) variable with the *imageAspectRatio*

            x = (i / self.W - 0.5) * 2 * half_width
            y = (j / self.H - 0.5) * 2 * half_height

            ray_origin = self.eye_pos
            ray_direct = Vec3(x, y, -1).normalized()

            pixel_color = self.ray_color(i, j, ray_origin, ray_direct)
            self.frame_buf[i, j] = pixel_color

    # Whitted-style RayTracer
    @ti.func
    def ray_color(self, i, j, origin, direction):
        pixel_color = Vec3(0)
        eps = 1e-5

        self.status_stack.clear(i, j)
        self.result_stack.clear(i, j)

        frame = StatusFrame(task_type=TASK_PROCESS, depth=0, ray_origin=origin, ray_direct=direction, kr=1.0)
        self.status_stack.push(i, j, frame)
        while not self.status_stack.empty(i, j):
            frame = self.status_stack.top(i, j)
            self.status_stack.pop(i, j)

            if frame.task_type == TASK_PROCESS:
                if frame.depth >= self.max_depth:
                    self.result_stack.push(i, j, Color(0, 0, 0))
                    continue

                is_hit, hit_t, hit_pos, hit_norm, hit_mat = self.world.hit(origin, direction)

                rd = frame.ray_direct.normalized()
                m_type = hit_mat.m_type
                depth = frame.depth

                if is_hit:
                    if m_type == REFLECTION_AND_REFRACTION or m_type == REFLECTION:
                        ref_idx = hit_mat.ior
                        cos_theta = clamp(-rd.dot(hit_norm), -1.0, 1.0)

                        front_face = cos_theta > 0
                        if front_face:
                            ref_idx = 1.0 / ref_idx

                        kr = reflectance(cos_theta, ref_idx)

                        reflect_direct = tm.reflect(rd, hit_norm).normalized()
                        reflect_origin = (
                            hit_pos + hit_norm * eps if reflect_direct.dot(hit_norm) > 0 else hit_pos - hit_norm * eps
                        )

                        merge_frame = StatusFrame(task_type=TASK_MERGE, depth=depth + 1, kr=kr)
                        self.status_stack.push(i, j, merge_frame)

                        reflect_frame = StatusFrame(
                            task_type=TASK_PROCESS,
                            depth=depth + 1,
                            ray_origin=reflect_origin,
                            ray_direct=reflect_direct,
                        )
                        self.status_stack.push(i, j, reflect_frame)

                        if m_type == REFLECTION_AND_REFRACTION:
                            refract_direct = tm.refract(rd, hit_norm, ref_idx).normalized()
                            refract_origin = (
                                hit_pos + hit_norm * eps
                                if refract_direct.dot(hit_norm) > 0
                                else hit_pos - hit_norm * eps
                            )

                            refract_frame = StatusFrame(
                                task_type=TASK_PROCESS,
                                depth=depth + 1,
                                ray_origin=refract_origin,
                                ray_direct=refract_direct,
                            )
                            self.status_stack.push(i, j, refract_frame)
                        else:  # m_type == REFLECTION
                            self.result_stack.push(i, j, Color(self.background_color))

                    else:
                        # [comment]
                        # We use the Phong illumation model int the default case. The phong model
                        # is composed of a diffuse and a specular reflection component.
                        # [/comment]
                        color = self.bling_phong(direction, hit_pos, hit_norm, hit_mat)
                        self.result_stack.push(i, j, color)
                else:
                    break

            elif frame.task_type == TASK_MERGE:
                kr = frame.kr
                refract_color = self.result_stack.top(i, j)
                self.result_stack.pop(i, j)
                reflect_color = self.result_stack.top(i, j)
                self.result_stack.pop(i, j)

                combined_color = kr * reflect_color + (1 - kr) * refract_color
                self.result_stack.push(i, j, combined_color)

        if not self.result_stack.empty(i, j):
            pixel_color = self.result_stack.top(i, j)
            self.result_stack.pop(i, j)
        else:
            pixel_color = self.background_color

        return pixel_color
