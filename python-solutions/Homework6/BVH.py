# Reference:
#   SAH: https://gitee.com/mmoon/taichi-rt/blob/master/ti_rt/utils/model.py
#   LBVH: https://forum.taichi-lang.cn/t/topic/4420/2

import taichi as ti
import numpy as np
from enum import Enum
from typing import Optional
from dataclasses import dataclass

from common import *
from Sorter import BitonicSortGPU


class SplitMethod(Enum):
    MIDDLE = 0
    SAH = 1
    LBVH = 2


class SorterType(Enum):
    QuickSort = 0
    BiSort = 1


@dataclass(slots=True)
class BVHNodeCPU:
    """Compact CPU node for SAH"""

    index: int = -1  # >= 0 when it is leaf, <0 when it is not leaf
    lBound: Vec3 = Vec3(Inf)
    uBound: Vec3 = Vec3(-Inf)
    lChild: int = -1
    rChild: int = -1
    parent: int = -1


@ti.dataclass
class BVHNodeGPU:
    """GPU node for LBVH"""

    index: ti.i32  # >= 0 when it is leaf, <0 when it is not leaf
    lBound: Vec3
    uBound: Vec3
    lChild: ti.i32
    rChild: ti.i32
    parent: ti.i32


def bound_diagonal(lower: NpArr, upper: NpArr) -> NpArr:
    return upper - lower


def bound_surface_area(lower: NpArr, upper: NpArr) -> float:
    d = bound_diagonal(lower, upper)
    return 2 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])


@ti.kernel
def duplicate_field(src_field: ti.template(), trg_field: ti.template()):
    for i in src_field:
        trg_field[i] = src_field[i]


@ti.pyfunc
def left_3_shift(x: ti.u32):
    # from pbrt
    if x == (1 << 10):
        x -= 1
    x = (x | (x << 16)) & 0b00000011000000000000000011111111
    # x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x | (x << 8)) & 0b00000011000000001111000000001111
    # x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x | (x << 4)) & 0b00000011000011000011000011000011
    # x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x | (x << 2)) & 0b00001001001001001001001001001001
    # x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x


@ti.data_oriented
class BVH:
    def __init__(self, vertices: NpArr, indices: NpArr, split_method=SplitMethod.MIDDLE, sorter=SorterType.QuickSort):
        self.primitives = vertices[indices].astype(np.float32)  # (primitive_count, 3, 3)
        self.primitive_count = len(indices)
        self.split_method = split_method
        self.sorter = sorter

        self.orders = np.arange(self.primitive_count)

    def build(self, split_method: Optional[SplitMethod] = None, sorter: Optional[SorterType] = None):
        if split_method:
            self.split_method = split_method
        if sorter:
            self.sorter = sorter

        match self.split_method:
            case SplitMethod.MIDDLE | SplitMethod.SAH:
                self.build_SAH()
            case SplitMethod.LBVH | _:
                self.build_LBVH()

    #################################################
    #################   Build SAH   #################
    #################################################

    # reference: https://gitee.com/mmoon/taichi-rt/blob/master/ti_rt/utils/model.py

    def build_SAH(self):
        self.nodes: list[BVHNodeCPU] = []
        self.nNodes = 0

        self.root = self.recursive_build(0, self.primitive_count)

        self.indices = ti.field(ti.i32, shape=self.nNodes)
        self.missNexts = ti.field(ti.i32, shape=self.nNodes)
        self.hitNexts = ti.field(ti.i32, shape=self.nNodes)
        self.lBounds = ti.Vector.field(3, ti.f32, shape=self.nNodes)
        self.uBounds = ti.Vector.field(3, ti.f32, shape=self.nNodes)

        missNexts = np.empty((self.nNodes), dtype=np.int32)
        missNexts[-1] = -1
        for i in range(self.nNodes - 2, -1, -1):
            parent = self.nodes[i].parent
            rChild = self.nodes[parent].rChild
            missNexts[i] = rChild if rChild != i else missNexts[parent]

        hitNexts = missNexts.copy()
        for i in range(self.nNodes):
            if self.nodes[i].lChild >= 0:
                hitNexts[i] = self.nodes[i].lChild
            elif self.nodes[i].rChild >= 0:
                hitNexts[i] = self.nodes[i].rChild

        self.missNexts.from_numpy(missNexts)
        self.hitNexts.from_numpy(hitNexts)
        self.indices.from_numpy(np.array([node.index for node in self.nodes], np.int32))
        self.lBounds.from_numpy(np.array([node.lBound for node in self.nodes], np.float32))
        self.uBounds.from_numpy(np.array([node.uBound for node in self.nodes], np.float32))

    def recursive_build(self, start: int, end: int) -> int:
        nPrimitives = end - start
        if nPrimitives == 1:
            obj = self.primitives[self.orders[start]]
            lBound, uBound = obj.min(axis=0), obj.max(axis=0)
            lChild, rChild = -1, -1
        elif nPrimitives == 2:
            lChild, rChild = len(self.nodes), len(self.nodes) + 1
            left_idx, right_idx = start, start + 1
            left_obj, right_obj = self.primitives[self.orders[left_idx]], self.primitives[self.orders[right_idx]]
            left_lBound, left_uBound = left_obj.min(axis=0), left_obj.max(axis=0)
            right_lBound, right_uBound = right_obj.min(axis=0), right_obj.max(axis=0)
            self.nodes.append(BVHNodeCPU(left_idx, left_lBound, left_uBound, -1, -1, lChild + 2))
            self.nodes.append(BVHNodeCPU(right_idx, right_lBound, right_uBound, -1, -1, lChild + 2))
            self.nNodes += 2

            lBound = np.minimum(left_lBound, right_lBound)
            uBound = np.maximum(left_uBound, right_uBound)
            start = -1  # not leaf
        else:
            objs = self.primitives[self.orders[start:end]]
            max_extent_axis = np.ptp(objs, axis=(0, 1)).argmax()
            sorted_indices = objs[:, :, max_extent_axis].sum(axis=1).argsort()
            self.orders[start:end] = self.orders[start:end][sorted_indices]

            match self.split_method:
                case SplitMethod.SAH:
                    mid = self.SAH_split(objs[sorted_indices], start, end)
                case SplitMethod.MIDDLE | _:
                    mid = start + (end - start) // 2

            lChild = self.recursive_build(start, mid)
            rChild = self.recursive_build(mid, end)
            self.nodes[lChild].parent = self.nodes[rChild].parent = len(self.nodes)
            lBound = np.minimum(self.nodes[lChild].lBound, self.nodes[rChild].lBound)
            uBound = np.maximum(self.nodes[lChild].uBound, self.nodes[rChild].uBound)
            start = -1  # not leaf
        self.nodes.append(BVHNodeCPU(start, lBound, uBound, lChild, rChild, -1))
        self.nNodes += 1
        return len(self.nodes) - 1

    def SAH_split(self, objs: NpArr, start: int, end: int) -> int:
        LOWERS = objs.min(axis=1)
        UPPERS = objs.max(axis=1)

        costs = np.zeros(end - start, np.float32)
        lMin = rMin = np.full(3, np.inf, np.float32)
        lMax = rMax = np.full(3, -np.inf, np.float32)
        for i in range(objs.shape[0]):
            j = objs.shape[0] - 1 - i
            if i > 0:
                costs[i] += bound_surface_area(lMin, lMax) * i
            lMin = np.minimum(lMin, LOWERS[i])
            lMax = np.maximum(lMax, UPPERS[i])
            rMin = np.minimum(rMin, LOWERS[j])
            rMax = np.maximum(rMax, UPPERS[j])
            costs[j] += bound_surface_area(rMin, rMax) * (i + 1)

        return min(int(start + max(1, costs.argmin())), end - 1)

    ##################################################
    #################   Build LBVH   #################
    ##################################################

    # reference: https://forum.taichi-lang.cn/t/topic/4420/2

    def build_LBVH(self):

        #### initialization ####

        self.internal_node_count = self.primitive_count - 1
        self.nodes = BVHNodeGPU.field(shape=(self.internal_node_count + self.primitive_count))
        self.codes = ti.field(ti.u32, shape=self.primitive_count)
        self.atomic_counter = ti.field(dtype=ti.i32, shape=[self.internal_node_count])

        lBounds = self.primitives.min(axis=1)
        uBounds = self.primitives.max(axis=1)
        self.min_bound = Vec3(lBounds.min(axis=0))
        self.max_bound = Vec3(uBounds.max(axis=0))

        self.lBounds = ti.Vector.field(3, ti.f32, shape=self.primitive_count)
        self.uBounds = ti.Vector.field(3, ti.f32, shape=self.primitive_count)
        self.lBounds.from_numpy(lBounds)
        self.uBounds.from_numpy(uBounds)

        self.root = 0
        self.nNodes = self.nodes.shape[0]

        self.missNexts = ti.field(ti.i32, shape=self.nNodes)
        self.hitNexts = ti.field(ti.i32, shape=self.nNodes)

        #### build tree ####

        # generate morton code
        self.generate_morton_code()
        codes = self.codes.to_numpy()

        # sort primitives by morton code
        match self.sorter:
            case SorterType.BiSort:
                self.orders = BitonicSortGPU(codes).argsort()
            case SorterType.QuickSort | _:
                self.orders = codes.argsort(kind="quicksort")

        codes = codes[self.orders]
        lBounds = lBounds[self.orders]
        uBounds = uBounds[self.orders]

        self.primitives = self.primitives[self.orders]
        self.codes.from_numpy(codes)
        self.lBounds.from_numpy(lBounds)
        self.uBounds.from_numpy(uBounds)

        # build nodes
        self.build_LBVH_nodes()
        self.set_primitive_idx()
        self.assign_bound()

        # flatten tree
        self.flatten_LBVH_missNexts()
        duplicate_field(self.missNexts, self.hitNexts)
        self.flatten_LBVH_hitNexts()

        #### post-processing ####

        self.indices = self.nodes.index
        self.lBounds = self.nodes.lBound
        self.uBounds = self.nodes.uBound

        self.codes = None
        self.atomic_counter = None

    @ti.kernel
    def generate_morton_code(self):
        for i in range(self.primitive_count):
            lb, ub = self.lBounds[i], self.uBounds[i]
            centroid = (lb + ub) * 0.5

            offset = centroid - self.min_bound
            x = offset * 1024 / (self.max_bound - self.min_bound)

            self.codes[i] = (
                left_3_shift(ti.floor(x[0], dtype=ti.i32))
                | left_3_shift(ti.floor(x[1], dtype=ti.i32)) << 1
                | left_3_shift(ti.floor(x[2], dtype=ti.i32)) << 2
            )

    @ti.pyfunc
    def delta(self, n1: ti.i32, n2: ti.i32):
        rv = -1
        if 0 <= n2 < self.primitive_count:
            c1 = self.codes[n1]
            c2 = self.codes[n2]
            v = 0
            if c1 == c2:
                c1 = n1
                c2 = n2
                v = 31

            c = c1 ^ c2
            rv = 31 - ti.floor(ti.log(c) / 0.69314, dtype=ti.i32) + v

        return rv

    @ti.kernel
    def build_LBVH_nodes(self):
        # from https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf

        self.nodes[0].parent = -1
        for i in range(self.internal_node_count):
            d = int(ti.math.sign(self.delta(i, i + 1) - self.delta(i, i - 1)))

            delta_min = self.delta(i, i - d)

            l_max = 2
            while self.delta(i, i + l_max * d) > delta_min:
                l_max = l_max * 2
            l = 0
            t = l_max // 2
            while t >= 1:
                if self.delta(i, i + (l + t) * d) > delta_min:
                    l = l + t
                t = t // 2
            j = i + l * d

            delta_node = self.delta(i, j)
            s = 0
            t = l
            div = 2
            while t > 1:
                # from floor to ceiling
                t = ti.ceil(l / div, dtype=ti.i32)
                if self.delta(i, i + (s + t) * d) > delta_node:
                    s = s + t
                div *= 2
            gamma = i + s * d + ti.min(d, 0)

            lChild = 0
            rChild = 0
            if ti.min(i, j) == gamma:
                lChild = gamma + self.internal_node_count
            else:
                lChild = gamma
            if ti.max(i, j) == gamma + 1:
                rChild = gamma + 1 + self.internal_node_count
            else:
                rChild = gamma + 1

            self.nodes[i].lChild = lChild
            self.nodes[i].rChild = rChild
            self.nodes[i].index = -1

            self.nodes[lChild].parent = i
            self.nodes[rChild].parent = i

    @ti.kernel
    def set_primitive_idx(self):
        for i in range(self.primitive_count):
            self.nodes[i + self.internal_node_count].index = i
            self.nodes[i + self.internal_node_count].lBound = self.lBounds[i]
            self.nodes[i + self.internal_node_count].uBound = self.uBounds[i]
            # self.nodes[i + self.internal_node_count].lChild = -1
            # self.nodes[i + self.internal_node_count].rChild = -1

    @ti.kernel
    def assign_bound(self):
        """
        as the original paper says:
        'Each thread starts from one leaf node and walks up the tree using parent pointers that we record during radix
        tree construction. We track how many threads have visited each internal node using atomic counters—the first
        thread terminates immediately while the second one gets to process the node. This way, each node is processed by
        exactly one thread, which leads to O(n) time complexity'
        """

        for i in range(self.primitive_count):
            idx = i + self.internal_node_count
            idx = self.nodes[idx].parent
            while idx >= 0:
                counter = ti.atomic_add(self.atomic_counter[idx], 1)
                if counter == 0:
                    idx = -1
                else:
                    lChild = self.nodes[idx].lChild
                    rChild = self.nodes[idx].rChild
                    lBound = ti.min(self.nodes[lChild].lBound, self.nodes[rChild].lBound)
                    uBound = ti.max(self.nodes[lChild].uBound, self.nodes[rChild].uBound)
                    self.nodes[idx].lBound = lBound
                    self.nodes[idx].uBound = uBound
                    idx = self.nodes[idx].parent

    @ti.kernel
    def flatten_LBVH_missNexts(self):
        self.missNexts[0] = 0
        ti.loop_config(serialize=True)
        for i in range(1, self.nNodes):
            parent = self.nodes[i].parent
            rChild = self.nodes[parent].rChild
            self.missNexts[i] = rChild if rChild != i else self.missNexts[parent]

    @ti.kernel
    def flatten_LBVH_hitNexts(self):
        for i in range(self.nNodes):
            lChild = self.nodes[i].lChild
            rChild = self.nodes[i].rChild
            if lChild >= 0:
                self.hitNexts[i] = lChild
            elif rChild >= 0:
                self.hitNexts[i] = rChild

    ##################################################
    #################   Algorithms   #################
    ##################################################

    @ti.pyfunc
    def hit_AABB(self, index, ray_origin, ray_direct, closest_t: float) -> bool:
        # invDir: ray direction(x,y,z), invDir=(1.0/x,1.0/y,1.0/z), use this because Multiply is faster that Division
        # dirIsNeg: ray direction(x,y,z), dirIsNeg=[int(x>0),int(y>0),int(z>0)], use this to simplify your logic
        # TODO: test if ray bound intersects

        i0 = (self.lBounds[index] - ray_origin) / ray_direct
        i1 = (self.uBounds[index] - ray_origin) / ray_direct
        return ti.max(ti.min(i0, i1).max(), 0) <= ti.min(ti.max(i0, i1).min(), closest_t)
