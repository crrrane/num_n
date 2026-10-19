def numbers(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [float(tok) for line in f for tok in line.split()]


def dimension(nums, path):
    if not nums:
        raise ValueError(f"input: {path}: empty file")
    n = int(nums[0])
    if float(n) != nums[0] or n <= 0:
        raise ValueError(
            f"input: {path}: first number must be integer dimension, found {nums[0]}"
        )
    return n


def read_dense_system(path):
    nums = numbers(path)
    n = dimension(nums, path)
    need = 1 + n * (n + 1)
    if len(nums) < need:
        raise ValueError(f"input: {path}: expected {need} numbers, found {len(nums)}")

    a = []
    b = []
    pos = 1
    for _ in range(n):
        a.append(nums[pos:pos + n])
        pos += n
        b.append(nums[pos])
        pos += 1
    return a, b


def read_tridiagonal(path):
    nums = numbers(path)
    n = dimension(nums, path)
    need = 1 + 4 * n
    if len(nums) < need:
        raise ValueError(f"input: {path}: expected {need} numbers, found {len(nums)}")

    lower = [0.0] * n
    main = [0.0] * n
    upper = [0.0] * n
    rhs = [0.0] * n

    pos = 1
    for i in range(n):
        lower[i] = nums[pos]
        main[i] = nums[pos + 1]
        upper[i] = nums[pos + 2]
        rhs[i] = nums[pos + 3]
        pos += 4

    return lower, main, upper, rhs


def read_iterative_system(path):
    a, b = read_dense_system(path)

    nums = numbers(path)
    n = dimension(nums, path)
    pos = 1 + n * (n + 1)

    if len(nums) <= pos:
        raise ValueError(f"input: {path}: expected precision after system")

    eps = nums[pos]
    if eps <= 0.0:
        raise ValueError(f"input: {path}: precision must be positive, found {eps}")

    return a, b, eps


def read_symmetric_matrix(path):
    nums = numbers(path)
    n = dimension(nums, path)
    need = 1 + n * n + 1
    if len(nums) < need:
        raise ValueError(f"input: {path}: expected {need} numbers, found {len(nums)}")

    a = []
    pos = 1
    for _ in range(n):
        a.append(nums[pos:pos + n])
        pos += n

    eps = nums[pos]
    if eps <= 0.0:
        raise ValueError(f"input: {path}: precision must be positive, found {eps}")

    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i][j] - a[j][i]) > 1e-12:
                raise ValueError(
                    f"input: {path}: matrix must be symmetric, "
                    f"a[{i + 1}][{j + 1}] != a[{j + 1}][{i + 1}]"
                )

    return a, eps


def read_square_matrix(path):
    nums = numbers(path)
    n = dimension(nums, path)
    need = 1 + n * n + 1
    if len(nums) < need:
        raise ValueError(f"input: {path}: expected {need} numbers, found {len(nums)}")
    a = []
    pos = 1
    for _ in range(n):
        a.append(nums[pos:pos+n])
        pos += n
    eps = nums[pos]
    if eps <= 0.0:
        raise ValueError(f"input: {path}: precision must be positive, found {eps}")
    return a, eps