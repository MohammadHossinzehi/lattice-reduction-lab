"""A short tour of lattice_lab. Run from the repository root:

    python examples/demo.py
"""

import random
import sys
import time
from decimal import Decimal, getcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lattice_lab import (bkz, hermite_factor, lll, minimal_polynomial,  # noqa: E402
                         shortest_vector, solve_integer_system)
from lattice_lab.__main__ import poly_str  # noqa: E402
from lattice_lab.linalg import norm2  # noqa: E402


def qary_lattice(n, k, q, seed):
    """Basis of {y : y = A x mod q} for a random n x (n - k) matrix A: the
    lattice family behind Learning With Errors. Rows: q e_i for i < k, then
    (a_j | e_j)."""
    rng = random.Random(seed)
    rows = [[q if j == i else 0 for j in range(n)] for i in range(k)]
    for i in range(k, n):
        rows.append([rng.randrange(q) for _ in range(k)] +
                    [1 if j == i else 0 for j in range(k, n)])
    return rows


def section(title):
    print()
    print(title)
    print("=" * len(title))


def main():
    section("1. LLL vs BKZ vs exact SVP on a 28 dimensional q-ary lattice")
    basis = qary_lattice(28, 14, 521, seed=7)
    print(f"input |b1|^2 = {norm2(basis[0])}")
    t = time.perf_counter()
    reduced, _ = lll(basis)
    t_lll = time.perf_counter() - t
    print(f"LLL      |b1|^2 = {norm2(reduced[0]):>6}  hermite factor "
          f"{hermite_factor(reduced):.4f}  ({t_lll:.2f}s)")
    for block in (4, 10):
        t = time.perf_counter()
        r, _ = bkz(basis, block)
        print(f"BKZ-{block:<4} |b1|^2 = {norm2(r[0]):>6}  hermite factor "
              f"{hermite_factor(r):.4f}  ({time.perf_counter() - t:.2f}s)")
    t = time.perf_counter()
    _, _, lam = shortest_vector(r)
    print(f"exact SVP lambda_1^2 = {lam}  ({time.perf_counter() - t:.2f}s)")

    section("2. Minimal polynomials from decimal digits")
    getcontext().prec = 100
    two, three, five = Decimal(2), Decimal(3), Decimal(5)
    cube_root_2 = (two.ln() / 3).exp()
    for name, alpha, deg in [
        ("sqrt(2) + sqrt(3)", two.sqrt() + three.sqrt(), 4),
        ("golden ratio", (1 + five.sqrt()) / 2, 2),
        ("2^(1/3) + 3^(1/2)", cube_root_2 + three.sqrt(), 6),
    ]:
        coeffs, res = minimal_polynomial(alpha, deg, digits=70)
        print(f"{name:<18} -> {poly_str(coeffs)}   (residual {res:.1e})")

    section("3. Every integer solution of a linear system")
    a = [[12, 18, -6, 30], [5, -7, 11, 3]]
    b = [36, 4]
    x0, kernel = solve_integer_system(a, b)
    print("A =", a, " b =", b)
    print("small particular solution:", x0)
    print("kernel lattice (LLL reduced):")
    for k in kernel:
        print("   ", k)


if __name__ == "__main__":
    main()
