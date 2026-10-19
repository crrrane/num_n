import sys
from lab_2.input import read_iterative_system

MAX_ITER = 100000


def show_vec(name, v, eps=1e-12):
    print(name)
    print(" ".join(f"{(0.0 if abs(x) < eps else x):10.4f}" for x in v))
    print()


def max_abs_diff(x, y):
    return max(abs(a - b) for a, b in zip(x, y))


def residual(a, b, x):
    n = len(a)
    mx = 0.0
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += a[i][j] * x[j]
        mx = max(mx, abs(s - b[i]))
    return mx


def gauss(a, b):
    n = len(a)
    m = [row[:] for row in a]
    y = b[:]

    for k in range(n):
        piv = k
        mx = abs(m[k][k])
        for i in range(k + 1, n):
            if abs(m[i][k]) > mx:
                mx = abs(m[i][k])
                piv = i

        if mx < 1e-15:
            raise ValueError("singular matrix")

        if piv != k:
            m[k], m[piv] = m[piv], m[k]
            y[k], y[piv] = y[piv], y[k]

        for i in range(k + 1, n):
            f = m[i][k] / m[k][k]
            for j in range(k, n):
                m[i][j] -= f * m[k][j]
            y[i] -= f * y[k]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= m[i][j] * x[j]
        x[i] = s / m[i][i]

    return x


def simple_iteration(a, b, eps):
    n = len(a)

    for i in range(n):
        if abs(a[i][i]) < 1e-15:
            raise ValueError(f"zero diagonal element at i={i + 1}")

    x = [0.0] * n

    for it in range(1, MAX_ITER + 1):
        x_new = [0.0] * n

        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= a[i][j] * x[j]
            x_new[i] = s / a[i][i]

        d = max_abs_diff(x_new, x)
        x = x_new

        if d < eps:
            return x, it, d

    raise ValueError(f"simple iteration did not converge in {MAX_ITER} iterations")


def seidel(a, b, eps):
    n = len(a)

    for i in range(n):
        if abs(a[i][i]) < 1e-15:
            raise ValueError(f"zero diagonal element at i={i + 1}")

    x = [0.0] * n

    for it in range(1, MAX_ITER + 1):
        d = 0.0

        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= a[i][j] * x[j]

            new = s / a[i][i]
            d = max(d, abs(new - x[i]))
            x[i] = new

        if d < eps:
            return x, it, d

    raise ValueError(f"Seidel did not converge in {MAX_ITER} iterations")


def run(path):
    a, b, eps = read_iterative_system(path)
    n = len(a)

    print(f"n = {n}")
    print(f"eps = {eps:.2e}\n")

    x_exact = gauss(a, b)
    show_vec("exact/reference x", x_exact)

    x_si, it_si, d_si = simple_iteration(a, b, eps)
    show_vec("simple iteration x", x_si)
    print(f"iterations = {it_si}")
    print(f"delta = {d_si:.2e}")
    print(f"residual = {residual(a, b, x_si):.2e}")
    print(f"error vs exact = {max_abs_diff(x_si, x_exact):.2e}\n")

    x_sd, it_sd, d_sd = seidel(a, b, eps)
    show_vec("Seidel x", x_sd)
    print(f"iterations = {it_sd}")
    print(f"delta = {d_sd:.2e}")
    print(f"residual = {residual(a, b, x_sd):.2e}")
    print(f"error vs exact = {max_abs_diff(x_sd, x_exact):.2e}\n")

    if it_si < it_sd:
        print(f"simple iteration is faster by factor {it_sd / it_si:.2f}")
    elif it_sd < it_si:
        print(f"Seidel is faster by factor {it_si / it_sd:.2f}")
    else:
        print("both methods require the same number of iterations")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data/lab1_3.txt")