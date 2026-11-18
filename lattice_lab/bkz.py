"""Block Korkine-Zolotarev (BKZ) reduction.

A BKZ tour walks a window of ``block_size`` vectors along the basis. For each
window [k, e) it solves SVP exactly in the projected sublattice
pi_k(L(b_k..b_{e-1})). If that projected shortest vector is noticeably
shorter than b*_k it is inserted at position k and LLL cleans up.

Insertion is done without ever creating a linearly dependent generating set.
The shortest vector's coefficient vector x is primitive (gcd 1), so a short
Euclid run of unimodular column operations on the window turns x into a unit
vector; the corresponding basis vector *is* the new vector and is rotated to
the front of the window. Every step is unimodular, so the transformation
matrix stays exact and invertible throughout.
"""

from fractions import Fraction

from .enumeration import projected_shortest
from .linalg import as_int_matrix, axpy, gram_schmidt, matmul
from .lll import lll_integral


def _insert(b, u, k, coeffs):
    """Make sum(coeffs[i] * b[k + i]) the new b[k], unimodularly."""
    x = list(coeffs)
    size = len(x)

    def col_op(i, j, q):
        # b_j <- b_j + q b_i  keeps v fixed if x_i <- x_i - q x_j
        b[k + j] = axpy(b[k + j], -q, b[k + i])
        u[k + j] = axpy(u[k + j], -q, u[k + i])
        x[i] -= q * x[j]

    while sum(1 for xi in x if xi) > 1:
        nz = [i for i in range(size) if x[i]]
        piv = min(nz, key=lambda i: abs(x[i]))
        for i in nz:
            if i != piv:
                col_op(i, piv, x[i] // x[piv])
    p = next(i for i in range(size) if x[i])
    if abs(x[p]) != 1:
        raise ArithmeticError("coefficient vector is not primitive")
    if x[p] == -1:
        b[k + p] = [-c for c in b[k + p]]
        u[k + p] = [-c for c in u[k + p]]
    for rows in (b, u):
        rows.insert(k, rows.pop(k + p))


def bkz(basis, block_size=10, delta=Fraction(99, 100), max_tours=50,
        stats=None):
    """BKZ-reduce ``basis``. Returns ``(reduced, U)`` with ``U * basis = reduced``.

    ``block_size = 2`` is essentially LLL; ``block_size = n`` gives an HKZ
    reduced basis whose first vector is a shortest lattice vector.
    """
    delta = Fraction(delta)
    b0 = as_int_matrix(basis)
    b, u = lll_integral(b0, delta)
    n = len(b)
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    tours = 0
    inserts = 0
    for tours in range(1, max_tours + 1):
        changed = False
        for k in range(n - 1):
            end = min(k + block_size, n)
            _, mu, B = gram_schmidt(b)
            x, length = projected_shortest(mu, B, k, end, delta * B[k])
            if x is None:
                continue
            _insert(b, u, k, x)
            b, u2 = lll_integral(b, delta)
            u = matmul(u2, u)
            changed = True
            inserts += 1
        if not changed:
            break
    if stats is not None:
        stats.update(tours=tours, inserts=inserts)
    return b, u


def hermite_factor(basis):
    """|b_1| / covol^(1/n): the usual quality measure for reduced bases."""
    from .linalg import gram_determinant, norm2
    n = len(basis)
    return (norm2(basis[0]) ** 0.5) / (gram_determinant(basis) ** (0.5 / n))
