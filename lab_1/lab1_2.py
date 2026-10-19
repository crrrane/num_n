import sys
from lab_2.input import read_tridiagonal

EPS = 1e-15


def show_vec(name, v, eps=1e-12):
    print(name)
    print(" ".join(f"{(0.0 if abs(x) < eps else x):10.4f}" for x in v))
    print()


def thomas(lower, main, upper, rhs):
    n = len(rhs)

    alpha = [0.0] * n
    beta = [0.0] * n

    if abs(main[0]) < EPS:
        raise ValueError("zero main diagonal element at i=1")

    den = main[0]
    alpha[0] = -upper[0] / den
    beta[0] = rhs[0] / den

    for i in range(1, n):
        den = main[i] + lower[i] * alpha[i - 1]
        if abs(den) < EPS:
            raise ValueError(f"zero denominator at step i={i + 1}")

        alpha[i] = -upper[i] / den
        beta[i] = (rhs[i] - lower[i] * beta[i - 1]) / den

    x = [0.0] * n
    x[-1] = beta[-1]

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return alpha, beta, x


def residual(lower, main, upper, rhs, x):
    n = len(rhs)
    mx = 0.0

    for i in range(n):
        s = main[i] * x[i]

        if i > 0:
            s += lower[i] * x[i - 1]

        if i + 1 < n:
            s += upper[i] * x[i + 1]

        mx = max(mx, abs(s - rhs[i]))

    return mx


def run(path):
    lower, main, upper, rhs = read_tridiagonal(path)

    if abs(lower[0]) > EPS:
        raise ValueError("a_1 must be 0 (lower[0] != 0)")
    if abs(upper[-1]) > EPS:
        raise ValueError("c_n must be 0 (upper[-1] != 0)")

    show_vec("a", lower)
    show_vec("b", main)
    show_vec("c", upper)
    show_vec("d", rhs)

    alpha, beta, x = thomas(lower, main, upper, rhs)

    show_vec("alpha", alpha)
    show_vec("beta", beta)
    show_vec("x", x)

    print(f"err = {residual(lower, main, upper, rhs, x):.2e}\n")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data/lab1_2.txt")