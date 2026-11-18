"""Things you can actually do with a lattice reducer.

* ``solve_integer_system``: every integer solution of A x = b, as a particular
  solution plus an LLL reduced kernel basis, with the particular solution
  shrunk by an exact closest vector search in the kernel lattice.
* ``integer_relation``: find small integers a with sum a_i x_i ~ 0 for given
  high precision reals (the LLL approach to PSLQ style problems).
* ``minimal_polynomial``: recover an algebraic number's minimal polynomial
  from its decimal expansion.
* ``simultaneous_approximation``: one denominator q that makes q * alpha_i
  close to integers for several alphas at once (Lagarias).
"""

from decimal import Decimal, localcontext
from fractions import Fraction
from math import gcd

from .enumeration import closest_vector
from .linalg import as_int_matrix, transpose
from .lll import lll_integral
from .normal_forms import smith_normal_form


class NoIntegerSolution(ValueError):
    pass


def solve_integer_system(a, b, shrink=True):
    """Solve A x = b over Z.

    Returns ``(x0, kernel)``: all solutions are x0 + sum k_i * kernel[i].
    Raises NoIntegerSolution when only rational (or no) solutions exist.
    """
    a = as_int_matrix(a)
    m, n = len(a), len(a[0])
    b = [int(x) for x in b]
    if len(b) != m:
        raise ValueError("b has the wrong length")
    s, u, v = smith_normal_form(a)
    ub = [sum(u[i][j] * b[j] for j in range(m)) for i in range(m)]
    r = sum(1 for i in range(min(m, n)) if s[i][i])
    y = [0] * n
    for i in range(r):
        if ub[i] % s[i][i]:
            raise NoIntegerSolution(
                f"invariant factor {s[i][i]} does not divide {ub[i]}: "
                "the system has no integer solution")
        y[i] = ub[i] // s[i][i]
    if any(ub[i] for i in range(r, m)):
        raise NoIntegerSolution("the system is inconsistent over Q")
    x0 = [sum(v[i][j] * y[j] for j in range(n)) for i in range(n)]
    vt = transpose(v)
    kernel = [vt[j] for j in range(r, n)]
    if kernel:
        kernel, _ = lll_integral(kernel)
        if shrink:
            close, _, _ = closest_vector(kernel, x0)
            x0 = [xi - ci for xi, ci in zip(x0, close)]
    return x0, kernel


def _to_decimal(x):
    if isinstance(x, Decimal):
        return x
    if isinstance(x, Fraction):
        return Decimal(x.numerator) / Decimal(x.denominator)
    return Decimal(str(x))


def integer_relation(values, digits=30, weight=None):
    """Find small integers a (not all zero) with sum a_i * values_i ~ 0.

    ``values`` should be Decimals or strings carrying at least ``digits``
    correct significant digits. Returns ``(a, residual)``.
    """
    with localcontext() as ctx:
        ctx.prec = digits + 20
        xs = [_to_decimal(v) for v in values]
        scale = Decimal(10) ** digits if weight is None else Decimal(weight)
        n = len(xs)
        rows = []
        for i in range(n):
            row = [1 if j == i else 0 for j in range(n)]
            row.append(int((xs[i] * scale).to_integral_value()))
            rows.append(row)
        reduced, _ = lll_integral(rows)
        a = reduced[0][:n]
        g = 0
        for c in a:
            g = gcd(g, c)
        a = [c // g for c in a]
        first = next(c for c in a if c)
        if first < 0:
            a = [-c for c in a]
        residual = sum(Decimal(c) * x for c, x in zip(a, xs))
    return a, residual


def minimal_polynomial(alpha, degree, digits=40):
    """Integer coefficients [c_0, ..., c_d] (lowest degree first) of a
    polynomial of degree <= ``degree`` vanishing at ``alpha``.

    ``alpha`` is a Decimal or string with plenty of digits. The returned
    polynomial is primitive with positive leading coefficient; trailing zero
    coefficients are dropped.
    """
    with localcontext() as ctx:
        ctx.prec = digits + 20
        x = _to_decimal(alpha)
        powers = [Decimal(1)]
        for _ in range(degree):
            powers.append(powers[-1] * x)
        coeffs, residual = integer_relation(powers, digits=digits)
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
    if coeffs and coeffs[-1] < 0:
        coeffs = [-c for c in coeffs]
    return coeffs, residual


def simultaneous_approximation(alphas, max_denominator):
    """Find q <= ~max_denominator and p_i so that |q alpha_i - p_i| is small
    for all i at once. Returns ``(q, [p_i], max_error)``.

    Uses the lattice with rows (1, C a_1, ..., C a_n) and C e_i, where C is
    chosen so that the denominator and the errors are balanced.
    """
    alphas = [Fraction(a) if not isinstance(a, (Decimal, str)) else
              Fraction(str(a)) for a in alphas]
    n = len(alphas)
    q_max = int(max_denominator)
    c = q_max ** (n + 1) if n else 1  # errors scaled to compete with q
    c = max(1, round(c ** (1.0 / n)))
    scale = Fraction(c)
    rows = [[1] + [round(scale * a) for a in alphas]]
    for i in range(n):
        rows.append([0] * (i + 1) + [c] + [0] * (n - i - 1))
    reduced, _ = lll_integral(rows)
    best = None
    for row in reduced:
        q = abs(row[0])
        if q == 0:
            continue
        ps = [round(q * a) for a in alphas]
        err = max(abs(q * a - p) for a, p in zip(alphas, ps))
        score = max(Fraction(q), c * err)  # the balanced sup norm
        if best is None or score < best[0]:
            best = (score, q, ps, err)
    return best[1], best[2], best[3]


def kernel_lattice(a):
    """An LLL reduced basis of {x in Z^n : A x = 0}."""
    _, kernel = solve_integer_system(a, [0] * len(a), shrink=False)
    return kernel


__all__ = ["solve_integer_system", "integer_relation", "minimal_polynomial",
           "simultaneous_approximation", "kernel_lattice", "NoIntegerSolution"]
