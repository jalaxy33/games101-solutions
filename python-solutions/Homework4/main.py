# Assignment 4. Bezier Curve

import os

os.chdir(os.path.dirname(__file__))

import time
import copy
import numpy as np
import taichi as ti
import taichi.math as tm


ti.init(arch=ti.gpu, default_fp=ti.f32, default_ip=ti.i32, random_seed=int(time.time()))

# define types
vec2 = ti.types.vector(2, float)
vec3 = ti.types.vector(3, float)
vec4 = ti.types.vector(4, float)


@ti.func
def factorial(n: int) -> int:
    tmp = 1
    for i in range(1, n + 1):
        tmp *= i
    return tmp


@ti.func
def combination(n: int, k: int) -> int:
    return factorial(n) / (factorial(k) * factorial(n - k))


# 参考：https://github.com/Zydiii/LearnTaiChi/blob/master/03/BezierBase.py
@ti.data_oriented
class Bezier:
    def __init__(self, N: int, width: int, height: int):
        # 阶数
        self.degree = N
        self.basePosNum = self.degree + 1
        self.t_num = 1000 * self.degree
        # image buffer
        self.W = width
        self.H = height
        self.image_buf = ti.Vector.field(3, float, shape=(self.W, self.H))
        # 基点坐标
        self.basePoint_pos = ti.Vector.field(2, float, shape=self.basePosNum)
        # 贝塞尔曲线坐标
        self.bezierCurve_pos = ti.Vector.field(2, float, shape=self.t_num)
        # 颜色
        self.basePoint_color = (0.8, 0.8, 0.8)
        self.curve_color = (1, 1, 0)

    # 设置随机端点坐标
    def setRandomBasePointPos(self):
        for i in range(0, self.basePosNum):
            # self.basePoint_pos[i] = ti.Vector([ti.sqrt(ti.random()) * 0.9, ti.sqrt(ti.random()) * 0.75])
            self.basePoint_pos[i] = ti.Vector([ti.sqrt(np.random.random()) * 0.9, ti.sqrt(np.random.random()) * 0.75])

        self.sortBasePoint()

    # 按照 x 坐标排序，便于可视化
    def sortBasePoint(self):
        base_positions = self.basePoint_pos.to_numpy()
        sorted_positions = base_positions[base_positions[:, 0].argsort()]
        self.basePoint_pos.from_numpy(sorted_positions)

    ##########################
    #      Naive Bezier      #
    ##########################

    # 计算贝塞尔曲线第 t 个点的位置
    @ti.pyfunc
    def _computeBezierPosition(self, t: int) -> vec2:
        # p(t) = Σ^n_{k = 0} p_k C(n,k) t^k (1-t)^(n - k)
        uStep = t / self.t_num
        point = vec2(0)
        for k in range(0, self.degree + 1):
            point += (
                self.basePoint_pos[k]
                * combination(self.degree, k)
                * tm.pow(uStep, k)
                * tm.pow(1 - uStep, self.degree - k)
            )
        return point

    # 计算贝塞尔曲线
    @ti.kernel
    def naive_bezier(self):
        for t in range(self.t_num):
            point = self._computeBezierPosition(t)
            self.bezierCurve_pos[t] = point

            x = int(point.x * self.W)
            y = int(point.y * self.H)
            self.image_buf[x, y] = self.curve_color

    @ti.kernel
    def reset(self):
        self.image_buf.fill(0)
        self.basePoint_pos.fill(0)
        self.bezierCurve_pos.fill(0)

    def display(self, window):
        # display curve
        canvas = window.get_canvas()
        canvas.set_image(self.image_buf)

        # display base points and lines
        indices = ti.field(int, shape=2 * self.degree)
        for i in range(0, self.degree):
            indices[2 * i] = i
            indices[2 * i + 1] = i + 1

        point_radius = 0.005
        canvas = window.get_canvas()
        canvas.circles(self.basePoint_pos, point_radius, self.basePoint_color)
        canvas.lines(self.basePoint_pos, point_radius / 5, indices, self.basePoint_color)


if __name__ == "__main__":
    # define display
    width = 1024
    height = 1024

    # compute bezier
    bezier = Bezier(N=4, width=width, height=height)
    bezier.setRandomBasePointPos()
    bezier.naive_bezier()

    # rendering
    window = ti.ui.Window("Draw Berzer Curve", (width, height))

    while window.running:
        if window.is_pressed(ti.ui.ESCAPE):  # 按ESC退出
            break

        if window.is_pressed(ti.ui.SPACE):  # 按空格重新生成
            bezier.reset()
            bezier.setRandomBasePointPos()
            bezier.naive_bezier()

        bezier.display(window)

        window.show()
