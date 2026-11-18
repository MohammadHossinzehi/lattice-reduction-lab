"""Exact shortest and closest vector search.

The core is one depth first Schnorr-Euchner style enumeration over the
Gram-Schmidt coordinates of the basis. At level i the remaining squared
budget R limits the coefficient x_i to |x_i - c_i| <= sqrt(R / B_i), where
c_i is the projected centre fixed by the coefficients already chosen above i.
Candidates are tried nearest-first, so the first leaf reached is usually
close to optimal and the budget shrinks quickly.

All comparisons are made on Fractions. Integer candidate ranges are obtained
with ``math.isqrt`` and then filtered exactly, so there is no floating point
anywhere and no epsilon to tune.
"""

from fractions import Fraction
from math import floor, isqrt

from .linalg import as_int_matrix, dot, gram_schmidt, norm2, round_half_up


def _candidates(c, radius2):
    """Integers x with (x - c)^2 < radius2 (radius2 > 0), nearest first."""
    t = isqrt(floor(radius2)) + 1
    base = floor(c)
    out = [x for x in range(base - t, base + t + 2) if (x - c) ** 2 < radius2]
    out.sort(key=lambda x: (abs(x - c), x))
    return out


def _enumerate(mu, B, start, end, tau, bound, allow_zero, stats=None):
    """Find integer x[start:end] minimising sum_i B_i (c_i - x_i)^2 < bound.

    ``tau`` holds target coordinates (zeros for SVP). Returns the best
    coefficient list and its squared length, or (None, bound) if nothing
    beats the bound.
    """
    size = end - start
    best = {"x": None, "len": Fraction(bound)}
    x = [0] * size
    nodes = [0]

    def rec(level, partial):
        i = start + level
        c = tau[i] - sum(x[j - start] * mu[j][i] for j in range(i + 1, end))
        room = (best["len"] - partial) / B[i]
        if room <= 0:
            return
        for xi in _candidates(c, room):
            nodes[0] += 1
            cost = partial + (c - xi) ** 2 * B[i]
            if cost >= best["len"]:
                break
            x[level] = xi
            if level == 0:
                if allow_zero or any(x):
                    best["x"] = list(x)
                    best["len"] = cost
            else:
                rec(level - 1, cost)
        x[level] = 0

    rec(size - 1, Fraction(0))
    if stats is not None:
        stats["nodes"] = nodes[0]
    return best["x"], best["len"]


def shortest_vector(basis, stats=None):
    """Return ``(v, coefficients, |v|^2)`` for a shortest nonzero vector.

    Exact for any basis. Reduce the basis with LLL first for speed: the
    search tree is exponentially smaller on a reduced basis.
    """
    b = as_int_matrix(basis)
    _, mu, B = gram_schmidt(b)
    n = len(b)
    # Start from the shortest basis vector; search for anything strictly shorter.
    first = min(range(n), key=lambda i: norm2(b[i]))
    best_coeffs = [1 if i == first else 0 for i in range(n)]
    bound = Fraction(norm2(b[first]))
    x, length = _enumerate(mu, B, 0, n, [Fraction(0)] * n, bound,
                           allow_zero=False, stats=stats)
    if x is not None:
        best_coeffs = x
        bound = length
    v = combine(b, best_coeffs)
    return v, best_coeffs, int(bound)


def projected_shortest(mu, B, start, end, bound):
    """Coefficients x of a vector whose projection orthogonal to b_1..b_start
    is shorter than ``bound`` and as short as possible, or None."""
    x, length = _enumerate(mu, B, start, end, [Fraction(0)] * len(B), bound,
                           allow_zero=False)
    return x, length


def babai_nearest_plane(basis, target):
    """Babai's nearest plane: a lattice vector close to ``target``.

    Guarantees |t - v|^2 <= (1/4) * sum_i B_i, so it is good on reduced bases.
    """
    b = as_int_matrix(basis)
    bstar, _, B = gram_schmidt(b)
    w = [Fraction(t) for t in target]
    coeffs = [0] * len(b)
    for i in range(len(b) - 1, -1, -1):
        c = dot(w, bstar[i]) / B[i]
        q = round_half_up(c)
        coeffs[i] = q
        w = [wi - q * bi for wi, bi in zip(w, b[i])]
    return combine(b, coeffs), coeffs


def closest_vector(basis, target, stats=None):
    """Exact closest vector problem: ``(v, coefficients, |t - v|^2)``.

    ``target`` may contain Fractions and need not lie in the span; the part
    orthogonal to the span is a constant added to every candidate distance.
    """
    b = as_int_matrix(basis)
    bstar, mu, B = gram_schmidt(b)
    t = [Fraction(x) for x in target]
    n = len(b)
    tau = [dot(t, bstar[i]) / B[i] for i in range(n)]
    perp = [ti - sum(tau[i] * bstar[i][k] for i in range(n))
            for k, ti in enumerate(t)]
    perp2 = norm2(perp)

    v0, c0 = babai_nearest_plane(b, t)
    d0 = norm2([ti - vi for ti, vi in zip(t, v0)])
    # Search strictly inside the Babai distance; ties keep Babai's answer.
    x, inside = _enumerate(mu, B, 0, n, tau, d0 - perp2, allow_zero=True,
                           stats=stats)
    if x is None:
        return v0, c0, d0
    v = combine(b, x)
    return v, x, inside + perp2


def combine(basis, coeffs):
    dim = len(basis[0])
    out = [0] * dim
    for c, row in zip(coeffs, basis):
        if c:
            for k in range(dim):
                out[k] += c * row[k]
    return out
