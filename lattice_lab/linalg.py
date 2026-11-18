"""Exact integer and rational linear algebra used by every other module.

Everything here works on plain Python lists of ints or Fractions. Nothing is
ever converted to a float, so results are exact no matter how large the
entries grow.
"""

from fractions import Fraction
from math import gcd

Matrix = list  # list of rows, each row a list of ints / Fractions


def as_int_matrix(rows):
    """Copy ``rows`` into a fresh list of lists of ints, checking shape."""
    out = [[int(x) for x in row] for row in rows]
    if out and any(len(r) != len(out[0]) for r in out):
        raise ValueError("all rows must have the same length")
    return out


def identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def norm2(v):
    return dot(v, v)


def matmul(a, b):
    if not a:
        return []
    cols = list(zip(*b))
    return [[dot(row, col) for col in cols] for row in a]


def transpose(a):
    return [list(col) for col in zip(*a)]


def axpy(y, q, x):
    """Return y - q * x (row operation helper)."""
    return [yi - q * xi for yi, xi in zip(y, x)]


def round_half_up(x):
    """Round a Fraction to the nearest integer, ties going up.

    The integral LLL uses (2*l + d) // (2*d); this is the same rule, which is
    what lets the rational and integral variants make identical decisions.
    """
    x = Fraction(x)
    return (2 * x.numerator + x.denominator) // (2 * x.denominator)


def det_bareiss(a):
    """Exact determinant of a square integer matrix (fraction free Bareiss)."""
    n = len(a)
    if n == 0:
        return 1
    m = [list(map(int, row)) for row in a]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if m[k][k] == 0:
            swap = next((i for i in range(k + 1, n) if m[i][k] != 0), None)
            if swap is None:
                return 0
            m[k], m[swap] = m[swap], m[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                m[i][j] = (m[i][j] * m[k][k] - m[i][k] * m[k][j]) // prev
        prev = m[k][k]
    return sign * m[n - 1][n - 1]


def gram_schmidt(basis):
    """Exact Gram-Schmidt orthogonalisation.

    Returns ``(bstar, mu, B)`` where ``bstar[i]`` are the orthogonal vectors,
    ``mu[i][j] = <b_i, b*_j> / B_j`` for j < i (with ``mu[i][i] = 1``) and
    ``B[i] = |b*_i|^2``. Raises ValueError if the rows are dependent.
    """
    n = len(basis)
    bstar = []
    mu = [[Fraction(0)] * n for _ in range(n)]
    B = []
    for i in range(n):
        v = [Fraction(x) for x in basis[i]]
        for j in range(i):
            mu[i][j] = Fraction(dot(basis[i], bstar[j])) / B[j]
            v = [vi - mu[i][j] * bj for vi, bj in zip(v, bstar[j])]
        mu[i][i] = Fraction(1)
        bi = norm2(v)
        if bi == 0:
            raise ValueError("basis vectors are linearly dependent")
        bstar.append(v)
        B.append(bi)
    return bstar, mu, B


def gram_determinant(basis):
    """det(B B^T): the squared covolume of the lattice spanned by the rows."""
    return det_bareiss(matmul(basis, transpose(basis)))


def is_unimodular(u):
    return len(u) == len(u[0]) and abs(det_bareiss(u)) == 1


def vector_gcd(v):
    g = 0
    for x in v:
        g = gcd(g, int(x))
    return g
