import sys
import math
from input import read_equation


SAFE = {
    "__builtins__": {},
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "tg": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "exp": math.exp, "sqrt": math.sqrt, "abs": abs,
    "log": math.log, "ln": math.log, "lg": math.log10,
    "pi": math.pi, "e": math.e,
}


def make_func(expr):
    code = compile(expr, "<eq>", "eval")
    def f(x):
        return float(eval(code, dict(SAFE), {"x": x}))
    return f


def make_deriv(f, h=1e-7):
    def df(x):
        return (f(x + h) - f(x - h)) / (2.0 * h)
    return df


def show(name):
    print(name)


def show_iter(k, x, fx, dx=None):
    if dx is None:
        print(f"{k:4d}   {x: .12f}   {fx: .6e}")
    else:
        print(f"{k:4d}   {x: .12f}   {fx: .6e}   {dx: .6e}")


def show_head(title):
    print(title)
    print(f"{'k':>4s}   {'x_k':>13s}   {'f(x_k)':>14s}   {'|dx|':>14s}")


def scan(f, a, b, step):
    n = max(1, int(round((b - a) / step)))
    h = (b - a) / n
    return [a + i * h for i in range(n + 1)]


def separate_roots(f, a, b, step):
    show("Graphical separation of roots")
    print(f"{'x':>12s}   {'f(x)':>16s}")
    xs = scan(f, a, b, step)
    fs = [f(x) for x in xs]
    intervals = []
    for i in range(len(xs)):
        mark = ""
        if i > 0 and fs[i - 1] * fs[i] < 0:
            mark = "   <-- sign change"
            intervals.append((xs[i - 1], xs[i]))
        print(f"{xs[i]:12.5f}   {fs[i]:16.6f}{mark}")
    print()
    return intervals


def bisection(f, a, b, eps, max_iter=200):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("no sign change on [a, b]")
    for it in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        if fc == 0.0 or (b - a) * 0.5 < eps:
            return c, it
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b), max_iter


def newton(f, df, x0, eps, max_iter=200):
    show_head("Newton's method")
    x = x0
    for k in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if dfx == 0.0:
            raise ZeroDivisionError("f'(x) = 0 at x = %g" % x)
        x_new = x - fx / dfx
        dx = abs(x_new - x)
        show_iter(k, x, fx, dx)
        if dx < eps:
            show_iter(k + 1, x_new, f(x_new), None)
            return x_new, k + 1
        x = x_new
    return x, max_iter


def simple_iteration(phi, f, x0, eps, max_iter=2000):
    show_head("Simple iteration method")
    x = x0
    for k in range(max_iter):
        x_new = phi(x)
        dx = abs(x_new - x)
        show_iter(k, x, f(x), dx)
        if dx < eps:
            show_iter(k + 1, x_new, f(x_new), None)
            return x_new, k + 1
        x = x_new
    return x, max_iter


def estimate_q(phi, a, b, n=400):
    h = (b - a) / n
    hp = 1e-6
    q = 0.0
    for i in range(n + 1):
        x = a + i * h
        d = (phi(x + hp) - phi(x - hp)) / (2.0 * hp)
        q = max(q, abs(d))
    return q


def run(path):
    eq = read_equation(path)
    f = make_func(eq["expr"])
    df = make_deriv(f)
    a, b = eq["a"], eq["b"]
    step, eps = eq["step"], eq["eps"]

    print(f"Equation:  f(x) = {eq['expr']} = 0")
    print(f"Interval:  [{a}, {b}]   scan step = {step}   eps = {eps}\n")

    intervals = separate_roots(f, a, b, step)
    if not intervals:
        print("No sign change found.")
        return
    print(f"Intervals with sign change: {len(intervals)}")
    for ia, ib in intervals:
        print(f"   [{ia:.5f}, {ib:.5f}]")
    print()

    positive = [(ia, ib) for ia, ib in intervals if ib > 0.0]
    if not positive:
        print("No positive root interval found.")
        return
    print(f"Searching positive roots in {len(positive)} interval(s)\n")

    for ia, ib in positive:
        print("=" * 62)
        print(f"Interval [{ia:.5f}, {ib:.5f}]")
        print("=" * 62)

        x0, it_bis = bisection(f, ia, ib, max(eps, 1e-3))
        print(f"Initial approximation (bisection): "
              f"x0 = {x0:.10f}, f(x0) = {f(x0):.3e}, iters = {it_bis}\n")

        x_newton, it_n = newton(f, df, x0, eps)
        print(f"Newton result:   x = {x_newton:.12f}, "
              f"f(x) = {f(x_newton):.3e}, iters = {it_n}\n")

        lam = 1.0 / df(x0)
        phi = (lambda lf: (lambda xx: xx - lf * f(xx)))(lam)
        q = estimate_q(phi, ia, ib)
        print(f"Simple iteration: phi(x) = x - lambda*f(x), "
              f"lambda = {lam:.6f}, q = {q:.4f}")
        if q >= 1.0:
            print("WARNING: q >= 1, convergence not guaranteed!")
        x_si, it_si = simple_iteration(phi, f, x0, eps)
        print(f"Simple iteration: x = {x_si:.12f}, "
              f"f(x) = {f(x_si):.3e}, iters = {it_si}\n")

        print("-" * 62)
        print(f"Summary for interval [{ia:.4f}, {ib:.4f}]:")
        print(f"  Newton:           {it_n:4d} iters   x = {x_newton:.10f}")
        print(f"  Simple iteration: {it_si:4d} iters   x = {x_si:.10f}")
        print()


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "data/lab2_1.txt")