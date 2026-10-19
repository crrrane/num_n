import sys
from lab_2.input import read_dense_system


def show(name, m, eps=1e-12):
    print(name)
    for row in m:
        print(" ".join(f"{(0.0 if abs(x) < eps else x):10.4f}" for x in row))
    print()


def show_vec(name, v, eps=1e-12):
    print(name)
    print(" ".join(f"{(0.0 if abs(x) < eps else x):10.4f}" for x in v))
    print()


def matmul(a, b):
    n, p, m = len(a), len(b), len(b[0])
    c = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            if a[i][k] == 0.0:
                continue
            for j in range(m):
                c[i][j] += a[i][k] * b[k][j]
    return c


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def lu(a):
    n = len(a)
    u = [row[:] for row in a]
    l = eye(n)
    p = eye(n)
    sw = 0

    for k in range(n):
        piv = k
        mx = abs(u[k][k])
        for i in range(k + 1, n):
            if abs(u[i][k]) > mx:
                mx = abs(u[i][k])
                piv = i
        if mx < 1e-15:
            raise ValueError("singular matrix")
        if piv != k:
            u[k], u[piv] = u[piv], u[k]
            p[k], p[piv] = p[piv], p[k]
            for j in range(k):
                l[k][j], l[piv][j] = l[piv][j], l[k][j]
            sw += 1
        for i in range(k + 1, n):
            f = u[i][k] / u[k][k]
            l[i][k] = f
            for j in range(k, n):
                u[i][j] -= f * u[k][j]
    return l, u, p, sw


def fwd(l, b):
    n = len(l)
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= l[i][j] * y[j]
        y[i] = s / l[i][i]
    return y


def bwd(u, y):
    n = len(u)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= u[i][j] * x[j]
        x[i] = s / u[i][i]
    return x


def solve(l, u, p, b):
    n = len(b)
    pb = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += p[i][j] * b[j]
        pb[i] = s
    return bwd(u, fwd(l, pb))


def inv(l, u, p):
    n = len(l)
    res = [[0.0] * n for _ in range(n)]
    for c in range(n):
        e = [0.0] * n
        e[c] = 1.0
        x = solve(l, u, p, e)
        for i in range(n):
            res[i][c] = x[i]
    return res


def det(u, sw):
    d = 1.0
    for i in range(len(u)):
        d *= u[i][i]
    return -d if sw % 2 else d


def diff(a, b):
    return max(abs(a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a[0])))


def run(path):
    a, b = read_dense_system(path)
    n = len(a)

    show("A", a)
    show_vec("b", b)

    l, u, p, sw = lu(a)
    show("L", l)
    show("U", u)
    show("P", p)

    lu_ = matmul(l, u)
    show("L*U", lu_)

    x = solve(l, u, p, b)
    show_vec("x", x)

    ainv = inv(l, u, p)
    show("A^-1", ainv)

    print(f"det(A) = {det(u, sw):.6f}\n")

    aa = matmul(a, ainv)
    show("A*A^-1", aa)
    print(f"err = {diff(aa, eye(n)):.2e}\n")

    pa = matmul(p, a)
    show("P*A", pa)
    show("L*U", lu_)
    print(f"err = {diff(lu_, pa):.2e}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data/lab1_1.txt")