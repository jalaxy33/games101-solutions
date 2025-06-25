import numpy as np
import taichi as ti
import taichi.math as tm


def taichi_solutions():
    # Basic Example of taichi
    print("\nExample of taichi:")
    a, b = 1.0, 2.0
    print("a =", a)
    print("a / b =", a / b)
    print("sqrt(b) =", tm.sqrt(b))
    print("acos(-1) =", tm.acos(-1))
    print("sin(30 / 180 * acos(-1)) =", tm.sin(30 / 180 * tm.acos(-1)))

    # Example of vector
    print("Example of vector:")
    v = tm.vec3(1.0, 2.0, 3.0)
    w = tm.vec3(1.0, 0.0, 0.0)
    print("v =", v)
    print("w =", w)
    print("v + w =", v + w)
    print("v * 3 =", v * 3)
    print("2 * v =", 2 * v)

    # Example of matrix
    print("Example of matrix:")
    i = tm.mat3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    j = tm.mat3(2.0, 3.0, 1.0, 4.0, 6.0, 5.0, 9.0, 7.0, 8.0)
    print("i =", i)
    print("j =", j)
    print("i + j =", i + j)
    print("i * 2.0 =", i * 2)
    print("i @ j =", i @ j)
    print("i @ v =", i @ v)


def numpy_solutions():
    print("\nExample of numpy:")
    a, b = 1.0, 2.0
    print("a =", a)
    print("a / b =", a / b)
    print("sqrt(b) =", np.sqrt(b))
    print("acos(-1) =", np.acos(-1))
    print("sin(30 / 180 * acos(-1)) =", np.sin(30 / 180 * np.acos(-1)))

    # Example of vector
    print("Example of vector:")
    v = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 0.0, 0.0])
    print("v =", v)
    print("w =", w)
    print("v + w =", v + w)
    print("v * 3 =", v * 3)
    print("2 * v =", 2 * v)

    # Example of matrix
    print("Example of matrix:")
    i = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    j = np.array([[2.0, 3.0, 1.0], [4.0, 6.0, 5.0], [9.0, 7.0, 8.0]])
    print("i =", i)
    print("j =", j)
    print("i + j =", i + j)
    print("i * 2.0 =", i * 2)
    print("i @ j =", i @ j)
    print("i @ v =", i @ v)


if __name__ == "__main__":
    ti.init(arch=ti.cpu)

    taichi_solutions()
    numpy_solutions()
