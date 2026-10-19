import sys
from math import sqrt
from lab_2.input import read_symmetric_matrix

MAX_ITER = 100000


def show(name, m, eps=1e-12):
    print(name)
    for row in m:
        print(" ".join(f"{(0.0 if abs(x) < eps else x):10.4f}" for x in row))
    print()


def show_vec(name, v, eps=1e-12):
    print(name)
    print(" ".join(f"{(0.0 if abs(x) < eps else x):10.4f}" for x in v))
    print()


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def diag_matrix(v):
    n = len(v)
    return [[v[i] if i == j else 0.0 for j in range(n)] for i in range(n)]


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


def max_abs_diff(a, b):
    return max(abs(a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a[0])))


def offdiag_max(a):
    n = len(a)
    mx = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            mx = max(mx, abs(a[i][j]))
    return mx


def offdiag_norm(a):
    n = len(a)
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += a[i][j] * a[i][j]
    return sqrt(2.0 * s)


def eigen_residual(a, v, lam):
    av = matmul(a, v)
    n = len(v)
    vlam = [[v[i][j] * lam[j] for j in range(n)] for i in range(n)]
    return max_abs_diff(av, vlam)


def jacobi(a, eps):
    n = len(a)

    if n == 0:
        raise ValueError("empty matrix")

    if n == 1:
        return [a[0][0]], [[1.0]], 0, [(0, 0.0, 0.0)]

    orig = [row[:] for row in a]
    a = [row[:] for row in a]
    v = eye(n)
    history = []

    for it in range(1, MAX_ITER + 1):
        p, q = 0, 1
        mx = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                val = abs(a[i][j])
                if val > mx:
                    mx = val
                    p, q = i, j

        if mx < eps:
            lam = [a[i][i] for i in range(n)]
            history.append((it - 1, mx, eigen_residual(orig, v, lam)))
            iterations = it - 1
            break

        app = a[p][p]
        aqq = a[q][q]
        apq = a[p][q]

        tau = (aqq - app) / (2.0 * apq)

        if tau >= 0.0:
            t = 1.0 / (tau + sqrt(1.0 + tau * tau))
        else:
            t = -1.0 / (-tau + sqrt(1.0 + tau * tau))

        c = 1.0 / sqrt(1.0 + t * t)
        s = t * c

        for k in range(n):
            if k != p and k != q:
                akp = a[k][p]
                akq = a[k][q]
                a[k][p] = c * akp - s * akq
                a[k][q] = s * akp + c * akq
                a[p][k] = a[k][p]
                a[q][k] = a[k][q]

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = 0.0
        a[q][p] = 0.0

        for k in range(n):
            vkp = v[k][p]
            vkq = v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq

        lam = [a[i][i] for i in range(n)]
        history.append((it, offdiag_max(a), eigen_residual(orig, v, lam)))

        if offdiag_max(a) < eps:
            iterations = it
            break
    else:
        raise ValueError(f"Jacobi method did not converge in {MAX_ITER} iterations")

    lam = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: lam[i])

    lam = [lam[i] for i in order]
    v = [[v[i][j] for j in order] for i in range(n)]

    return lam, v, iterations, history


def run(path):
    a, eps = read_symmetric_matrix(path)
    n = len(a)

    show("A", a)
    print(f"eps = {eps:.2e}\n")

    lam, v, iterations, history = jacobi(a, eps)

    show_vec("eigenvalues", lam)
    show("eigenvectors V (columns)", v)
    show("Lambda", diag_matrix(lam))

    av = matmul(a, v)
    vlam = [[v[i][j] * lam[j] for j in range(n)] for i in range(n)]

    show("A*V", av)
    show("V*Lambda", vlam)
    print(f"err = {max_abs_diff(av, vlam):.2e}\n")

    print("iteration history:")
    print(" iter      max_offdiag      eig_residual")
    for it, off, res in history:
        print(f"{it:5d} {off:18.6e} {res:18.6e}")
    print()

    print(f"iterations = {iterations}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data/lab1_4.txt")