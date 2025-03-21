# Reference: https://gitee.com/mmoon/taichi-rt/blob/master/ti_rt/utils/model.py

import taichi as ti
import taichi.math as tm
import numpy as np
from enum import Enum

from dataclasses import dataclass
from typing import *

from common import *


class SplitMethod(Enum):
    MIDDLE = 0
    SAH = 1


@dataclass(slots=True)
class BVHNode:
    index: int  # 节点序号
    lBound: Vec3
    uBound: Vec3
    left: int  # 左子节点
    right: int  # 右子节点
    parent: int


# Normal BVH Tree
@ti.data_oriented
class BVHTree:
    def __init__(self, primitives, split_method=SplitMethod.SAH):
        self.primitives = np.array(primitives, np.float32)
        self.nodes: list[BVHNode] = []
        self.order = np.arange(len(primitives))
        self.split_method = split_method

    def build(self, split_method: Optional[SplitMethod] = None):
        if split_method:
            self.split_method = split_method

        self.recursive_build(0, len(self.primitives))

        # After building trees

        missNext = np.empty((len(self.nodes),), dtype=np.int32)
        missNext[-1] = -1
        for i in range(len(self.nodes) - 2, -1, -1):
            parent = self.nodes[i].parent
            rc = self.nodes[parent].right
            missNext[i] = rc if rc != i else missNext[parent]

        hitNext = missNext.copy()
        for i in range(len(self.nodes)):
            if self.nodes[i].left >= 0:
                hitNext[i] = self.nodes[i].left
            elif self.nodes[i].right >= 0:
                hitNext[i] = self.nodes[i].right

        n = len(self.nodes)
        self.nNodes = n
        self.index = ti.field(dtype=ti.i32, shape=n)
        self.missNext = ti.field(dtype=ti.i32, shape=n)
        self.hitNext = ti.field(dtype=ti.i32, shape=n)
        self.lBound = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.uBound = ti.Vector.field(3, dtype=ti.f32, shape=n)

        self.missNext.from_numpy(missNext)
        self.hitNext.from_numpy(hitNext)
        self.index.from_numpy(np.array([node.index for node in self.nodes], np.int32))
        self.lBound.from_numpy(np.array([node.lBound for node in self.nodes], np.float32))
        self.uBound.from_numpy(np.array([node.uBound for node in self.nodes], np.float32))

    def recursive_build(self, begin: int, end: int) -> int:
        if end - begin == 1:
            left = right = -1
            lb = self.primitives[self.order[begin]].min(axis=0)
            ub = self.primitives[self.order[begin]].max(axis=0)
        elif end - begin == 2:
            left, right = len(self.nodes), len(self.nodes) + 1
            left_lb = self.primitives[self.order[begin]].min(axis=0)
            left_ub = self.primitives[self.order[begin]].max(axis=0)
            right_lb = self.primitives[self.order[begin + 1]].min(axis=0)
            right_ub = self.primitives[self.order[begin + 1]].max(axis=0)
            self.nodes.append(BVHNode(begin, left_lb, left_ub, -1, -1, left + 2))
            self.nodes.append(BVHNode(begin + 1, right_lb, right_ub, -1, -1, left + 2))
            lb = np.minimum(left_lb, right_lb)
            ub = np.maximum(left_ub, right_ub)
            begin = -1
        else:
            _order = self.order[begin:end]
            _prims = self.primitives[_order]
            __ord = _prims[:, :, np.ptp(_prims, axis=(0, 1)).argmax()].sum(axis=1).argsort()
            self.order[begin:end] = _order[__ord]
            match self.split_method:
                case SplitMethod.SAH:
                    cost = np.zeros(end - begin, dtype=np.float32)
                    self.SAH_split(_prims[__ord], cost)
                    mid = min(int(begin + max(1, cost.argmin())), end - 1)
                case SplitMethod.MIDDLE | _:
                    mid = (begin + end) // 2
            left = self.recursive_build(begin, mid)
            right = self.recursive_build(mid, end)
            self.nodes[left].parent = self.nodes[right].parent = len(self.nodes)
            lb = np.minimum(self.nodes[left].lBound, self.nodes[right].lBound)
            ub = np.maximum(self.nodes[left].uBound, self.nodes[right].uBound)
            begin = -1
        self.nodes.append(BVHNode(begin, lb, ub, left, right, -1))
        return len(self.nodes) - 1

    @ti.kernel
    def SAH_split(self, primitives: ti.types.ndarray(Mat3x3), cost: ti.types.ndarray(ti.f32)):
        """Surface Area Heuristic 表面积启发式分割"""
        lMin = rMin = Vec3(Inf, Inf, Inf)
        lMax = rMax = -Vec3(Inf, Inf, Inf)
        ti.loop_config(serialize=True)
        for i in range(primitives.shape[0]):
            j = primitives.shape[0] - 1 - i
            if i > 0:
                lDx = lMax - lMin
                cost[i] += (lDx.x * lDx.y + lDx.x * lDx.z + lDx.y * lDx.z) * i
            lMin = min(lMin, [primitives[i][:, k].min() for k in range(3)])
            lMax = max(lMax, [primitives[i][:, k].max() for k in range(3)])
            rMin = min(rMin, [primitives[j][:, k].min() for k in range(3)])
            rMax = max(rMax, [primitives[j][:, k].max() for k in range(3)])
            rDx = rMax - rMin
            cost[j] += (rDx.x * rDx.y + rDx.x * rDx.z + rDx.y * rDx.z) * (i + 1)

    @ti.func
    def hitAABB(self, id: int, ray: Ray3, tmin: float) -> bool:
        # invDir: ray direction(x,y,z), invDir=(1.0/x,1.0/y,1.0/z), use this because Multiply is faster that Division
        # dirIsNeg: ray direction(x,y,z), dirIsNeg=[int(x>0),int(y>0),int(z>0)], use this to simplify your logic
        # TODO: test if ray bound intersects

        i1 = (self.lBound[id] - ray.origin) / ray.direct
        i2 = (self.uBound[id] - ray.origin) / ray.direct
        return tm.max(tm.min(i1, i2).max(), 0) <= tm.min(tm.max(i1, i2).min(), tmin)
