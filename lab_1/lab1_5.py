import sys
import math
import cmath
from lab_2.input import read_square_matrix

MAX_ITER = 100000


def show(name, m, eps=1e-12):
    print(name)
    for row in m:
        print(" ".join(f"{(0.0 if abs(x) < eps else x):10.4f}" for x in row))
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


def qr_decompose(a):
    n = len(a)
    q = eye(n)
    r = [row[:] for row in a]
    for k in range(n - 1):
        x = [r[i][k] for i in range(k, n)]
        norm_x = math.sqrt(sum(v * v for v in x))
        if norm_x < 1e-15:
            continue
        alpha = -norm_x if x[0] >= 0 else norm_x
        v = x[:]
        v[0] -= alpha
        norm_v = math.sqrt(sum(vv * vv for vv in v))
        if norm_v < 1e-15:
            continue
        v = [vv / norm_v for vv in v]
        for j in range(k, n):
            s = sum(v[i - k] * r[i][j] for i in range(k, n))
            for i in range(k, n):
                r[i][j] -= 2.0 * v[i - k] * s
        for i in range(n):
            s = sum(q[i][j] * v[j - k] for j in range(k, n))
            for j in range(k, n):
                q[i][j] -= 2.0 * s * v[j - k]
    return q, r


def qr_decompose_complex(a):
    n = len(a)
    q = [[complex(1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    r = [[a[i][j] for j in range(n)] for i in range(n)]
    for k in range(n - 1):
        x = [r[i][k] for i in range(k, n)]
        norm_x = math.sqrt(sum(abs(v) ** 2 for v in x))
        if norm_x < 1e-15:
            continue
        if abs(x[0]) < 1e-15:
            phase = 1.0
        else:
            phase = x[0] / abs(x[0])
        alpha = -phase * norm_x
        v = x[:]
        v[0] -= alpha
        norm_v = math.sqrt(sum(abs(vv) ** 2 for vv in v))
        if norm_v < 1e-15:
            continue
        v = [vv / norm_v for vv in v]
        for j in range(k, n):
            s = sum(v[m - k].conjugate() * r[m][j] for m in range(k, n))
            for i in range(k, n):
                r[i][j] -= 2.0 * v[i - k] * s
        for i in range(n):
            s = sum(q[i][m] * v[m - k] for m in range(k, n))
            for j in range(k, n):
                q[i][j] -= 2.0 * s * v[j - k].conjugate()
    return q, r


def qr_algorithm(a, eps, max_iter=MAX_ITER):
    n = len(a)
    a = [[complex(a[i][j]) for j in range(n)] for i in range(n)]
    eigvals = []
    m = n
    it_total = 0
    while m > 1:
        for it in range(max_iter):
            it_total += 1
            if abs(a[m - 1][m - 2]) < eps:
                eigvals.append(a[m - 1][m - 1])
                m -= 1
                break
            a11 = a[m - 2][m - 2]
            a12 = a[m - 2][m - 1]
            a21 = a[m - 1][m - 2]
            a22 = a[m - 1][m - 1]
            trace = a11 + a22
            det = a11 * a22 - a12 * a21
            disc = cmath.sqrt(trace * trace - 4 * det)
            l1 = (trace + disc) / 2
            l2 = (trace - disc) / 2
            shift = l1 if abs(l1 - a22) < abs(l2 - a22) else l2
            shifted = [[a[i][j] - (shift if i == j else 0) for j in range(m)] for i in range(m)]
            q, r = qr_decompose_complex(shifted)
            a_block = matmul(r, q)
            for i in range(m):
                a_block[i][i] += shift
            for i in range(m):
                for j in range(m):
                    a[i][j] = a_block[i][j]
        else:
            raise ValueError("QR algorithm did not converge")
    if m == 1:
        eigvals.append(a[0][0])
    return eigvals, it_total


def print_eigenvalues(eigvals, eps=1e-8):
    for val in eigvals:
        if abs(val.imag) < eps:
            print(f"  {val.real:.6f}")
        else:
            print(f"  {val.real:.6f} + {val.imag:.6f}i")


def run(path):
    a, eps = read_square_matrix(path)

    show("A", a)
    print(f"eps = {eps:.2e}\n")

    q, r = qr_decompose(a)
    show("Q", q)
    show("R", r)
    show("Q*R", matmul(q, r))

    eigvals, it = qr_algorithm(a, eps)
    print(f"Eigenvalues (QR algorithm, {it} iterations):")
    print_eigenvalues(eigvals)
    print()

    print("Example with complex conjugate eigenvalues")
    a_ex = [[1.0, 0.0, 0.0],
            [0.0, 2.0, -1.0],
            [0.0, 1.0, 2.0]]
    show("A_ex", a_ex)
    eigvals_ex, it_ex = qr_algorithm(a_ex, 1e-10)
    print(f"Eigenvalues (our QR algorithm, {it_ex} iterations):")
    print_eigenvalues(eigvals_ex)
    print()

    try:
        import numpy as np
        np_eigvals = np.linalg.eigvals(np.array(a_ex))
        print("NumPy check:")
        for val in np_eigvals:
            print(f"  {val.real:.6f} + {val.imag:.6f}i")
    except ImportError:
        print("NumPy is not installed, reference check skipped.")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data/lab1_5.txt")