"""LLL lattice basis reduction, two exact implementations.

``lll_rational`` keeps the Gram-Schmidt data as Fractions and updates it
incrementally on every size reduction and swap (Cohen, Algorithm 2.6.3).

``lll_integral`` is de Weger's fraction free variant (Cohen, Algorithm
2.6.7): it never stores a Fraction, only the integers d_i = det Gram(b_1..b_i)
and lambda_{k,j} = d_{j+1} * mu_{k,j}. All divisions in it are exact.

Both follow the same schedule (reduce against k-1, test Lovasz, swap or
finish size reducing), use the same rounding rule, and make the same
comparison in exact arithmetic. They therefore perform the *same* sequence of
operations and return identical bases, which the test suite checks.

Both return ``(reduced_basis, U)`` with ``U * input == reduced_basis`` and U
unimodular.
"""

from fractions import Fraction

from .linalg import as_int_matrix, axpy, gram_schmidt, identity, round_half_up


def _check_delta(delta):
    delta = Fraction(delta)
    if not Fraction(1, 4) < delta <= 1:
        raise ValueError("delta must satisfy 1/4 < delta <= 1")
    return delta


def lll_rational(basis, delta=Fraction(99, 100), stats=None):
    b = as_int_matrix(basis)
    delta = _check_delta(delta)
    n = len(b)
    u = identity(n)
    if n <= 1:
        return b, u
    _, mu, B = gram_schmidt(b)
    swaps = 0

    def reduce(k, l):
        q = round_half_up(mu[k][l])
        if q == 0 or abs(mu[k][l]) <= Fraction(1, 2):
            return
        b[k] = axpy(b[k], q, b[l])
        u[k] = axpy(u[k], q, u[l])
        mu[k][l] -= q
        for i in range(l):
            mu[k][i] -= q * mu[l][i]

    k = 1
    while k < n:
        reduce(k, k - 1)
        if B[k] < (delta - mu[k][k - 1] ** 2) * B[k - 1]:
            swaps += 1
            m = mu[k][k - 1]
            new_prev = B[k] + m * m * B[k - 1]
            mu[k][k - 1] = m * B[k - 1] / new_prev
            B[k] = B[k - 1] * B[k] / new_prev
            B[k - 1] = new_prev
            b[k], b[k - 1] = b[k - 1], b[k]
            u[k], u[k - 1] = u[k - 1], u[k]
            for j in range(k - 1):
                mu[k][j], mu[k - 1][j] = mu[k - 1][j], mu[k][j]
            for i in range(k + 1, n):
                t = mu[i][k]
                mu[i][k] = mu[i][k - 1] - m * t
                mu[i][k - 1] = t + mu[k][k - 1] * mu[i][k]
            k = max(1, k - 1)
        else:
            for l in range(k - 2, -1, -1):
                reduce(k, l)
            k += 1
    if stats is not None:
        stats["swaps"] = swaps
    return b, u


def lll_integral(basis, delta=Fraction(99, 100), stats=None):
    b = as_int_matrix(basis)
    delta = _check_delta(delta)
    p, q_ = delta.numerator, delta.denominator
    n = len(b)
    u = identity(n)
    if n <= 1:
        return b, u

    # 1-indexed bookkeeping: d[0] = 1, d[i] for i = 1..n; lam[k][j] for j < k.
    d = [1] + [0] * n
    lam = [[0] * (n + 1) for _ in range(n + 1)]
    for k in range(1, n + 1):
        for j in range(1, k + 1):
            acc = sum(x * y for x, y in zip(b[k - 1], b[j - 1]))
            for i in range(1, j):
                acc = (d[i] * acc - lam[k][i] * lam[j][i]) // d[i - 1]
            if j < k:
                lam[k][j] = acc
            else:
                d[k] = acc
        if d[k] == 0:
            raise ValueError("basis vectors are linearly dependent")
    swaps = 0

    def reduce(k, l):
        if 2 * abs(lam[k][l]) <= d[l]:
            return
        q = (2 * lam[k][l] + d[l]) // (2 * d[l])
        b[k - 1] = axpy(b[k - 1], q, b[l - 1])
        u[k - 1] = axpy(u[k - 1], q, u[l - 1])
        lam[k][l] -= q * d[l]
        for i in range(1, l):
            lam[k][i] -= q * lam[l][i]

    k = 2
    while k <= n:
        reduce(k, k - 1)
        # Lovasz: B_k < (delta - mu^2) B_{k-1}, multiplied through by
        # q * d_{k-1}^2 / ... to stay in the integers.
        lhs = q_ * d[k] * d[k - 2]
        rhs = p * d[k - 1] ** 2 - q_ * lam[k][k - 1] ** 2
        if lhs < rhs:
            swaps += 1
            b[k - 1], b[k - 2] = b[k - 2], b[k - 1]
            u[k - 1], u[k - 2] = u[k - 2], u[k - 1]
            for j in range(1, k - 1):
                lam[k][j], lam[k - 1][j] = lam[k - 1][j], lam[k][j]
            lm = lam[k][k - 1]
            new_d = (d[k - 2] * d[k] + lm * lm) // d[k - 1]
            for i in range(k + 1, n + 1):
                t = lam[i][k]
                lam[i][k] = (d[k] * lam[i][k - 1] - lm * t) // d[k - 1]
                lam[i][k - 1] = (new_d * t + lm * lam[i][k]) // d[k]
            d[k - 1] = new_d
            k = max(2, k - 1)
        else:
            for l in range(k - 2, 0, -1):
                reduce(k, l)
            k += 1
    if stats is not None:
        stats["swaps"] = swaps
    return b, u


def lll(basis, delta=Fraction(99, 100), method="integral"):
    """Reduce ``basis`` (rows) and return ``(reduced, U)``."""
    if method == "integral":
        return lll_integral(basis, delta)
    if method == "rational":
        return lll_rational(basis, delta)
    raise ValueError("method must be 'integral' or 'rational'")


def is_size_reduced(basis):
    _, mu, _ = gram_schmidt(basis)
    return all(abs(mu[i][j]) <= Fraction(1, 2)
               for i in range(len(basis)) for j in range(i))


def is_lll_reduced(basis, delta=Fraction(99, 100)):
    """Check both LLL conditions exactly."""
    delta = Fraction(delta)
    _, mu, B = gram_schmidt(basis)
    n = len(basis)
    for i in range(n):
        for j in range(i):
            if abs(mu[i][j]) > Fraction(1, 2):
                return False
    for k in range(1, n):
        if B[k] < (delta - mu[k][k - 1] ** 2) * B[k - 1]:
            return False
    return True
