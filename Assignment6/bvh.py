import taichi as ti
import taichi.math as tm
import numpy as np
from enum import Enum
from typing import Optional
from dataclasses import dataclass

Inf = float("inf")
Vec3 = ti.types.vector(3, dtype=ti.f32)
Mat3x3 = ti.types.matrix(3, 3, dtype=ti.f32)


class SplitMethod(Enum):
    MIDDLE = 0
    SAH = 1


@dataclass(slots=True)
class BVHNode:
    index: int = -1
    lBound: Vec3 = Vec3(Inf)
    uBound: Vec3 = Vec3(-Inf)
    left: int = -1
    right: int = -1
    parent: int = -1


@ti.data_oriented
class BVHTree:
    def __init__(self, primitives_np, split_method=SplitMethod.SAH):
        self.primitives = np.array(primitives_np, dtype=np.float32)  # (count, 3, 3)
        self.order = np.arange(len(self.primitives))
        self.nodes: list[BVHNode] = []
        self.split_method = split_method

        self.root = -1  # nNode - 1

    def build(self, split_method: Optional[SplitMethod] = None):
        if split_method:
            self.split_method = split_method

        # build tree
        self.root = self.recursive_build(0, len(self.primitives))

        # after building tree
        n = len(self.nodes)
        self.nNodes = n

        self.index = ti.field(dtype=ti.i32, shape=n)
        self.missNext = ti.field(dtype=ti.i32, shape=n)
        self.hitNext = ti.field(dtype=ti.i32, shape=n)
        self.lBound = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.uBound = ti.Vector.field(3, dtype=ti.f32, shape=n)

        missNext = np.empty((n,), dtype=np.int32)
        missNext[-1] = -1
        for i in range(len(self.nodes) - 2, -1, -1):
            parent = self.nodes[i].parent
            right = self.nodes[parent].right
            missNext[i] = right if right != i else missNext[parent]

        hitNext = missNext.copy()
        for i in range(len(self.nodes)):
            if self.nodes[i].left >= 0:
                hitNext[i] = self.nodes[i].left
            elif self.nodes[i].right >= 0:
                hitNext[i] = self.nodes[i].right

        self.missNext.from_numpy(missNext)
        self.hitNext.from_numpy(hitNext)
        self.index.from_numpy(np.array([node.index for node in self.nodes], np.int32))
        self.lBound.from_numpy(np.array([node.lBound for node in self.nodes], np.float32))
        self.uBound.from_numpy(np.array([node.uBound for node in self.nodes], np.float32))

    def recursive_build(self, begin, end) -> int:
        if end - begin == 1:
            left = right = -1
            lbound = self.primitives[self.order[begin]].min(axis=0)
            ubound = self.primitives[self.order[begin]].max(axis=0)
        elif end - begin == 2:
            left, right = len(self.nodes), len(self.nodes) + 1
            left_lbound = self.primitives[self.order[begin]].min(axis=0)
            left_ubound = self.primitives[self.order[begin]].max(axis=0)
            right_lbound = self.primitives[self.order[begin + 1]].min(axis=0)
            right_ubound = self.primitives[self.order[begin + 1]].max(axis=0)
            self.nodes.append(BVHNode(begin, left_lbound, left_ubound, -1, -1, left + 2))
            self.nodes.append(BVHNode(begin + 1, right_lbound, right_ubound, -1, -1, left + 2))
            lbound = np.minimum(left_lbound, right_lbound)
            ubound = np.maximum(left_ubound, right_ubound)
            begin = -1
        else:
            _order = self.order[begin:end]
            _objs = self.primitives[_order]
            __ord = _objs[:, :, np.ptp(_objs, axis=(0, 1)).argmax()].sum(axis=1).argsort()
            self.order[begin:end] = _order[__ord]
            match self.split_method:
                case SplitMethod.SAH:
                    cost = np.zeros(end - begin, dtype=np.float32)
                    self.SAH_split(_objs[__ord], cost)
                    mid = min(int(begin + max(1, cost.argmin())), end - 1)
                case SplitMethod.MIDDLE | _:
                    mid = (begin + end) // 2
            left = self.recursive_build(begin, mid)
            right = self.recursive_build(mid, end)
            self.nodes[left].parent = self.nodes[right].parent = len(self.nodes)
            lbound = np.minimum(self.nodes[left].lBound, self.nodes[right].lBound)
            ubound = np.maximum(self.nodes[left].uBound, self.nodes[right].uBound)
            begin = -1
        self.nodes.append(BVHNode(begin, lbound, ubound, left, right, -1))
        return len(self.nodes) - 1

    @ti.kernel
    def SAH_split(self, objs: ti.types.ndarray(Mat3x3), cost: ti.types.ndarray(ti.f32)):
        """Surface Area Heuristic 表面积启发式分割"""
        lMin = rMin = Vec3(Inf)
        lMax = rMax = Vec3(-Inf)
        ti.loop_config(serialize=True)
        for i in range(objs.shape[0]):
            j = objs.shape[0] - 1 - i
            if i > 0:
                lDx = lMax - lMin
                cost[i] += (lDx.x * lDx.y + lDx.x * lDx.z + lDx.y * lDx.z) * i
            lMin = min(lMin, [objs[i][:, k].min() for k in range(3)])
            lMax = max(lMax, [objs[i][:, k].max() for k in range(3)])
            rMin = min(rMin, [objs[j][:, k].min() for k in range(3)])
            rMax = max(rMax, [objs[j][:, k].max() for k in range(3)])
            rDx = rMax - rMin
            cost[j] += (rDx.x * rDx.y + rDx.x * rDx.z + rDx.y * rDx.z) * (i + 1)

    @ti.pyfunc
    def hit_AABB(self, index, ray_origin, ray_direct, closest_t: float) -> bool:
        # invDir: ray direction(x,y,z), invDir=(1.0/x,1.0/y,1.0/z), use this because Multiply is faster that Division
        # dirIsNeg: ray direction(x,y,z), dirIsNeg=[int(x>0),int(y>0),int(z>0)], use this to simplify your logic
        # TODO: test if ray bound intersects

        i0 = (self.lBound[index] - ray_origin) / ray_direct
        i1 = (self.uBound[index] - ray_origin) / ray_direct

        return tm.max(tm.min(i0, i1).max(), 0) <= tm.min(tm.max(i0, i1).min(), closest_t)
