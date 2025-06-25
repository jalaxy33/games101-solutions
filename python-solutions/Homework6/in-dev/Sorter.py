# Reference:
#   bitonic-sort: https://forum.taichi-lang.cn/t/taichi/1370
#                 https://gitee.com/real_french_fries/taichi_bitonic-sort

import taichi as ti

from common import *


def next_power_of_two(n: int) -> int:
    if n <= 0:
        return 1
    bit_len = n.bit_length()
    return 1 << (bit_len - 1) if (1 << (bit_len - 1)) == n else 1 << bit_len


@ti.pyfunc
def xor(a, b):
    return (a + b) & 1


@ti.data_oriented
class BitonicSortGPU:
    """
    Reference:
    - https://forum.taichi-lang.cn/t/taichi/1370
    - https://gitee.com/real_french_fries/taichi_bitonic-sort
    """

    def __init__(self, values: NpArr):
        self.n0 = len(values)  # original len
        self.n = next_power_of_two(len(values))

        self.step = ti.field(ti.i32, 2)
        self.step[0] = 2  # outer step size
        self.step[1] = 2  # inner step size

        self.values0 = values.astype(np.float64)  # original values

        self.values = ti.field(ti.f64)
        self.orders = ti.field(ti.i32)
        ti.root.dense(ti.i, self.n).place(self.values, self.orders)

        self.init_values(self.values0)
        self.init_orders()

        self.sorted = False

    @ti.kernel
    def init_values(self, values: ti.types.ndarray(ti.f64)):
        self.values.fill(Inf)
        for i in range(values.shape[0]):
            self.values[i] = values[i]

    @ti.kernel
    def init_orders(self):
        for i in range(self.orders.shape[0]):
            self.orders[i] = i

    @ti.kernel
    def bit_sort(self):
        for i in range(self.n // 2):
            stepLength = ti.Vector([self.step[0], self.step[1]])
            halfstep = stepLength[1] // 2
            i1 = i + (i // halfstep) * halfstep
            i2 = i1 + halfstep

            order1, order2 = self.orders[i1], self.orders[i2]
            a, b = self.values[i1], self.values[i2]
            updown = (i * 2 // stepLength[0]) & 1
            if xor(a > b, updown):
                self.orders[i1], self.orders[i2] = order2, order1
                self.values[i1], self.values[i2] = b, a

    @ti.kernel
    def inplace_bit_sort(self):
        for i in range(self.n // 2):
            stepLength = ti.Vector([self.step[0], self.step[1]])
            halfstep = stepLength[1] // 2
            i1 = i + (i // halfstep) * halfstep
            i2 = i1 + halfstep
            # order1, order2 = self.orders[i1], self.orders[i2]
            a, b = self.values[i1], self.values[i2]
            updown = (i * 2 // stepLength[0]) & 1
            if xor(a > b, updown):
                # self.orders[i1], self.orders[i2] = order2, order1
                self.values[i1], self.values[i2] = b, a

    def get_results(self):
        return self.values.to_numpy()[: self.n0]

    def get_orders(self):
        return self.orders.to_numpy()[: self.n0]

    def sort(self):
        if not self.sorted:
            for i in range(1, self.n.bit_length()):
                self.step[0] = 2**i
                for j in range(i):
                    self.step[1] = 2 ** (i - j)
                    self.inplace_bit_sort()
        self.sorted = True
        return self.get_results()

    def argsort(self):
        if self.sorted:
            self.init_values(self.values0.copy())
            self.init_orders()

        for i in range(1, self.n.bit_length()):
            self.step[0] = 2**i
            for j in range(i):
                self.step[1] = 2 ** (i - j)
                self.bit_sort()
        self.sorted = True
        return self.get_orders()


if __name__ == "__main__":

    ti.init(ti.gpu)

    n2 = 25
    n = 2**n2
    n -= 2
    values = ti.field(ti.f32, n)

    @ti.kernel
    def init():
        for i in range(n):
            values[i] = ti.random(ti.f32)

    init()
    print("Initial values:", values)

    values_cpu = values.to_numpy()

    sorter = BitonicSortGPU(values_cpu)
    results = sorter.sort()
    orders = sorter.argsort()
    print("BiSort results: ", results)
    print("BiSort orders:", orders)

    print("BiSort results2: ", sorter.sort())
    print("BiSort orders2:", sorter.argsort())

    values_copied = values_cpu.copy()
    values_copied.sort(kind="quicksort")
    orders_np = values_cpu.argsort(kind="quicksort")
    print("Numpy results: ", values_copied)
    print("Numpy orders:", orders_np)
