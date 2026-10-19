import sys
import math
from input import read_system


SAFE = {
    "__builtins__": {},
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "tg": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "exp": math.exp, "sqrt": math.sqrt, "abs": abs,
    "log": math.log, "ln": math.log, "lg": math.log10,
    "pi": math.pi, "e": math.e,
}


def make_funcs(exprs, n, a):
    codes = [compile(expr, "<f>", "eval") for expr in exprs]

    def f(x):
        local = {f"x{i+1}": x[i] for i in range(n)}
        local["a"] = a
        return [float(eval(code, SAFE, local)) for code in codes]

    return f


def make_phi(exprs, n, a):
    codes = [compile(expr, "<phi>", "eval") for expr in exprs]

    def phi(x):
        local = {f"x{i+1}": x[i] for i in range(n)}
        local["a"] = a
        return [float(eval(code, SAFE, local)) for code in codes]

    return phi


def jacobian(f, x):
    n = len(x)
    J = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            h = 1e-6 * max(1.0, abs(x[j]))
            x_plus = x[:]
            x_minus = x[:]
            x_plus[j] += h
            x_minus[j] -= h
            f_plus = f(x_plus)
            f_minus = f(x_minus)
            J[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * h)
    return J


def solve_linear(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-15:
            raise ValueError("singular matrix")
        M[col], M[pivot] = M[pivot], M[col]

        for r in range(col + 1, n):
            factor = M[r][col] / M[col][col]
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def norm_inf(v):
    return max(abs(x) for x in v)


def show_head(title, n):
    print(title)
    header = f"{'k':>4s}"
    for i in range(n):
        header += f"   {'x'+str(i+1):>13s}"
    for i in range(n):
        header += f"   {'f'+str(i+1):>14s}"
    header += f"   {'||dx||':>14s}"
    print(header)


def show_iter(k, x, fx, dx):
    line = f"{k:4d}"
    for v in x:
        line += f"   {v: .12f}"
    for v in fx:
        line += f"   {v: .6e}"
    if dx is not None:
        line += f"   {dx: .6e}"
    print(line)


def newton_system(f, x0, eps, max_iter):
    show_head("Newton's method for system", len(x0))
    x = x0[:]
    for k in range(max_iter):
        fx = f(x)
        J = jacobian(f, x)
        delta = solve_linear(J, [-v for v in fx])
        x_new = [x[i] + delta[i] for i in range(len(x))]
        dx = norm_inf(delta)
        show_iter(k, x, fx, dx)
        if dx < eps:
            show_iter(k + 1, x_new, f(x_new), None)
            return x_new, k + 1
        x = x_new
    return x, max_iter


def simple_iteration(phi, f, x0, eps, max_iter):
    show_head("Simple iteration method", len(x0))
    x = x0[:]
    for k in range(max_iter):
        x_new = phi(x)
        fx = f(x_new)
        dx = norm_inf([x_new[i] - x[i] for i in range(len(x))])
        show_iter(k, x, fx, dx)
        if dx < eps:
            show_iter(k + 1, x_new, f(x_new), None)
            return x_new, k + 1
        x = x_new
    return x, max_iter


def estimate_q_phi(phi, x0, n):
    J = jacobian(phi, x0)
    q = max(sum(abs(J[i][j]) for j in range(n)) for i in range(n))
    return q


def run(path):
    sys_data = read_system(path)
    n = sys_data["n"]
    a = sys_data["a"]
    eps = sys_data["eps"]
    max_iter = sys_data["max_iter"]
    x0 = sys_data["x0"]

    f = make_funcs(sys_data["f_exprs"], n, a)
    phi = make_phi(sys_data["phi_exprs"], n, a)

    print(f"System of {n} nonlinear equations")
    print(f"Parameter a = {a}")
    print(f"Initial approximation: {x0}")
    print(f"eps = {eps}, max_iter = {max_iter}\n")

    # Newton
    x_newton, it_n = newton_system(f, x0, eps, max_iter)
    print(f"\nNewton result: x = {x_newton}, f(x) = {f(x_newton)}, iters = {it_n}\n")

    # Simple iteration
    q = estimate_q_phi(phi, x0, n)
    print(f"Simple iteration: estimated q = {q:.4f}")
    if q >= 1.0:
        print("WARNING: q >= 1, convergence not guaranteed!")
    x_si, it_si = simple_iteration(phi, f, x0, eps, max_iter)
    print(f"\nSimple iteration result: x = {x_si}, f(x) = {f(x_si)}, iters = {it_si}\n")

    print("Summary:")
    print(f"  Newton:           {it_n:4d} iters   x = {x_newton}")
    print(f"  Simple iteration: {it_si:4d} iters   x = {x_si}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data/lab2_2.txt")