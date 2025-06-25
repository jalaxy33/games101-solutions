import taichi as ti
import numpy as np


Inf = float("Inf")
Vec3 = ti.types.vector(3, dtype=ti.f32)
NpArr = np.ndarray


class TailRecurseException(BaseException):
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


def tail_call_optimized(g):
    """
    Python 尾递归优化修饰器

    测试代码：
    @tail_call_optimized
    def fib(n, a, b):
        if n == 1:
            return a
        else:
            return fib(n-1, b, a+b)

    print(fib(1200, 0, 1))
    """
    import sys

    def func(*args, **kwargs):
        f = sys._getframe()
        # 如果产生新的递归调用栈帧时
        if f.f_back and f.f_back.f_back and f.f_back.f_back.f_code == f.f_code:
            # 捕获当前尾调用函数的参数，并抛出异常
            raise TailRecurseException(args, kwargs)
        else:
            while 1:
                try:
                    return g(*args, **kwargs)
                except TailRecurseException as e:
                    args = e.args
                    kwargs = e.kwargs

    func.__doc__ = g.__doc__
    return func
