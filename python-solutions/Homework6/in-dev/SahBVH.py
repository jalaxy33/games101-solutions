import taichi as ti
import numpy as np
from queue import Queue
from dataclasses import dataclass, field

from common import *

InfArr3 = np.full(3, np.inf, np.float32)


@dataclass(slots=True)
class Bound:
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    @classmethod
    def default(cls):
        return Bound(lower=(Inf, Inf, Inf), upper=(-Inf, -Inf, -Inf))

    def union(self, b: "Bound"):
        lower = tuple(np.minimum(self.lower, b.lower).tolist())
        upper = tuple(np.maximum(self.upper, b.upper).tolist())
        return Bound(lower, upper)


@dataclass(slots=True)
class BVHNode:
    is_leaf: bool = True
    bound: Bound = field(default_factory=Bound.default)
    left_child: "BVHNode" = None
    right_child: "BVHNode" = None
    parent: "BVHNode" = None


class SahBVH:
    def __init__(self, vertices: NpArr, indices: NpArr):
        self.vertices = vertices
        self.indices = indices

        self.primitives = vertices[indices].astype(np.float32)
        self.primitive_count = len(indices)

        self.lower_bounds = self.primitives.min(axis=1)
        self.upper_bounds = self.primitives.max(axis=1)
        self.centroids = (self.lower_bounds + self.upper_bounds) * 0.5


    def build(self):
        self.nodes = []

        

