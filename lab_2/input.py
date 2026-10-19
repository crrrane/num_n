def read_equation(path):
    data = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    if "expr" not in data or "a" not in data or "b" not in data:
        raise ValueError("file must contain keys: expr, a, b")
    return {
        "expr": data["expr"],
        "a": float(data["a"]),
        "b": float(data["b"]),
        "step": float(data.get("step", 0.1)),
        "eps": float(data.get("eps", 1e-6)),
    }


def read_system(path):
    data = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()

    if "n" not in data:
        raise ValueError("file must contain key: n")
    if "x0" not in data:
        raise ValueError("file must contain key: x0")

    n = int(data["n"])
    a = float(data.get("a", 0.0))
    eps = float(data.get("eps", 1e-6))
    max_iter = int(data.get("max_iter", 200))
    x0 = [float(v) for v in data["x0"].split(",")]

    f_exprs = []
    phi_exprs = []
    for i in range(n):
        f_key = f"f{i+1}"
        phi_key = f"phi{i+1}"
        if f_key not in data:
            raise ValueError(f"file must contain key: {f_key}")
        if phi_key not in data:
            raise ValueError(f"file must contain key: {phi_key}")
        f_exprs.append(data[f_key])
        phi_exprs.append(data[phi_key])

    return {
        "n": n,
        "a": a,
        "eps": eps,
        "max_iter": max_iter,
        "x0": x0,
        "f_exprs": f_exprs,
        "phi_exprs": phi_exprs,
    }