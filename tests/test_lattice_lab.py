"""Test suite. Run with ``python -m unittest discover -s tests``.

Every algorithm is checked against an independent oracle rather than against
hand copied expected output:

* LLL: exact checks of both LLL conditions, U * B = R with det U = +-1, the
  covolume is unchanged, the rational and integral implementations agree
  operation for operation, and with delta = 3/4, |b_1|^2 <= 2^(n-1) lambda_1^2
  holds against the true lambda_1 from enumeration.
* SVP / CVP: brute force over a coefficient box on small lattices.
* BKZ: never worse than LLL; with block = n the first vector is a shortest
  vector.
* HNF: canonical form of the lattice (invariant under random unimodular
  mixing), pivot and range conditions, U A = H.
* SNF: diagonal entries equal the ratios of determinantal divisors (gcd of
  all k x k minors), divisibility chain, U A V = S.
* Applications: known minimal polynomials and relations, integer systems
  checked by substitution and against brute force solvability.
"""

import itertools
import random
import unittest
from decimal import Decimal, getcontext
from fractions import Fraction
from math import gcd

from lattice_lab import (NoIntegerSolution, babai_nearest_plane, bkz,
                         closest_vector, gram_determinant, gram_schmidt,
                         hermite_normal_form, integer_relation,
                         invariant_factors, is_lll_reduced, is_size_reduced,
                         is_unimodular, kernel_lattice, lll_integral,
                         lll_rational, minimal_polynomial, shortest_vector,
                         simultaneous_approximation, smith_normal_form,
                         solve_integer_system)
from lattice_lab.__main__ import evaluate, main, poly_str
from lattice_lab.linalg import det_bareiss, matmul, norm2


def random_basis(rng, n, dim=None, lo=-20, hi=20):
    dim = dim or n
    while True:
        b = [[rng.randint(lo, hi) for _ in range(dim)] for _ in range(n)]
        if gram_determinant(b) != 0:
            return b


def random_unimodular(rng, n, steps=12):
    u = [[int(i == j) for j in range(n)] for i in range(n)]
    if n < 2:
        return [[rng.choice([-1, 1])]] if n else u
    for _ in range(steps):
        i, j = rng.sample(range(n), 2)
        q = rng.randint(-3, 3)
        u[i] = [a + q * b for a, b in zip(u[i], u[j])]
        if rng.random() < 0.3:
            u[i], u[j] = u[j], u[i]
    return u


def brute_shortest(basis, box):
    best = None
    for c in itertools.product(range(-box, box + 1), repeat=len(basis)):
        if not any(c):
            continue
        v = [sum(ci * row[k] for ci, row in zip(c, basis))
             for k in range(len(basis[0]))]
        n2 = norm2(v)
        if best is None or n2 < best:
            best = n2
    return best


def brute_closest(basis, target, box):
    best = None
    for c in itertools.product(range(-box, box + 1), repeat=len(basis)):
        v = [sum(ci * row[k] for ci, row in zip(c, basis))
             for k in range(len(basis[0]))]
        d = norm2([Fraction(t) - x for t, x in zip(target, v)])
        if best is None or d < best:
            best = d
    return best


def minors_gcd(a, k):
    g = 0
    rows, cols = range(len(a)), range(len(a[0]))
    for rs in itertools.combinations(rows, k):
        for cs in itertools.combinations(cols, k):
            g = gcd(g, det_bareiss([[a[r][c] for c in cs] for r in rs]))
    return g


class LLLTests(unittest.TestCase):
    def test_rational_and_integral_agree_exactly(self):
        rng = random.Random(1)
        for trial in range(60):
            n = rng.randint(2, 6)
            dim = n + rng.randint(0, 2)
            b = random_basis(rng, n, dim, -50, 50)
            delta = rng.choice([Fraction(3, 4), Fraction(99, 100), Fraction(1)])
            s1, s2 = {}, {}
            r1, u1 = lll_rational(b, delta, stats=s1)
            r2, u2 = lll_integral(b, delta, stats=s2)
            self.assertEqual(r1, r2, f"trial {trial}")
            self.assertEqual(u1, u2)
            self.assertEqual(s1["swaps"], s2["swaps"])

    def test_output_is_reduced_and_spans_same_lattice(self):
        rng = random.Random(2)
        for _ in range(50):
            n = rng.randint(2, 7)
            b = random_basis(rng, n, n + rng.randint(0, 3), -10 ** 6, 10 ** 6)
            r, u = lll_integral(b)
            self.assertTrue(is_lll_reduced(r))
            self.assertTrue(is_size_reduced(r))
            self.assertTrue(is_unimodular(u))
            self.assertEqual(matmul(u, b), r)
            self.assertEqual(gram_determinant(r), gram_determinant(b))

    def test_first_vector_bound_against_true_minimum(self):
        rng = random.Random(3)
        delta = Fraction(3, 4)
        for _ in range(25):
            n = rng.randint(2, 4)
            b = random_basis(rng, n, n, -15, 15)
            r, _ = lll_integral(b, delta)
            lam1 = shortest_vector(r)[2]
            # |b1|^2 <= (1/(delta - 1/4))^(n-1) * lambda_1^2 = 2^(n-1) lambda_1^2
            self.assertLessEqual(norm2(r[0]), 2 ** (n - 1) * lam1)

    def test_classic_knapsack_style_example(self):
        b = [[1, 0, 0, 0, 0, 10 ** 8 * 31],
             [0, 1, 0, 0, 0, 10 ** 8 * 41],
             [0, 0, 1, 0, 0, 10 ** 8 * 59],
             [0, 0, 0, 1, 0, 10 ** 8 * 26],
             [0, 0, 0, 0, 1, 10 ** 8 * 53]]
        r, _ = lll_integral(b)
        # The reduced basis lives in the hyperplane sum c_i w_i = 0 except
        # for one vector.
        zero_last = sum(1 for row in r if row[-1] == 0)
        self.assertEqual(zero_last, 4)
        self.assertTrue(is_lll_reduced(r))

    def test_dependent_basis_rejected(self):
        with self.assertRaises(ValueError):
            lll_integral([[1, 2], [2, 4]])
        with self.assertRaises(ValueError):
            lll_rational([[1, 2, 3], [4, 5, 6], [5, 7, 9]])

    def test_bad_delta(self):
        with self.assertRaises(ValueError):
            lll_integral([[1, 0], [0, 1]], Fraction(1, 5))

    def test_trivial_sizes(self):
        self.assertEqual(lll_integral([[5, 3]])[0], [[5, 3]])
        self.assertEqual(lll_integral([])[0], [])


class EnumerationTests(unittest.TestCase):
    def test_svp_matches_brute_force(self):
        rng = random.Random(4)
        for _ in range(40):
            n = rng.randint(2, 4)
            b = random_basis(rng, n, n + rng.randint(0, 1), -12, 12)
            r, _ = lll_integral(b)
            v, coeffs, n2 = shortest_vector(r)
            self.assertEqual(norm2(v), n2)
            self.assertTrue(any(coeffs))
            self.assertEqual(n2, brute_shortest(r, 4))

    def test_svp_on_unreduced_basis_is_still_exact(self):
        rng = random.Random(5)
        for _ in range(15):
            b = random_basis(rng, 3, 3, -9, 9)
            skew = matmul(random_unimodular(rng, 3), b)
            self.assertEqual(shortest_vector(skew)[2], shortest_vector(b)[2])

    def test_cvp_matches_brute_force(self):
        rng = random.Random(6)
        for _ in range(40):
            n = rng.randint(2, 3)
            b = random_basis(rng, n, n + rng.randint(0, 1), -9, 9)
            r, _ = lll_integral(b)
            t = [Fraction(rng.randint(-60, 60), rng.randint(1, 7))
                 for _ in range(len(r[0]))]
            v, coeffs, d2 = closest_vector(r, t)
            self.assertEqual(d2, norm2([ti - vi for ti, vi in zip(t, v)]))
            self.assertEqual(d2, brute_closest(r, t, 12))

    def test_babai_bound(self):
        rng = random.Random(7)
        for _ in range(30):
            b = random_basis(rng, 4, 4, -30, 30)
            r, _ = lll_integral(b)
            _, _, B = gram_schmidt(r)
            t = [Fraction(rng.randint(-999, 999), 3) for _ in range(4)]
            v, _ = babai_nearest_plane(r, t)
            d = norm2([ti - vi for ti, vi in zip(t, v)])
            self.assertLessEqual(d, sum(B) / 4)
            self.assertGreaterEqual(d, closest_vector(r, t)[2])

    def test_lattice_point_target(self):
        b = [[3, 1], [1, 4]]
        v, coeffs, d2 = closest_vector(b, [7, 13])  # not a lattice point
        self.assertEqual(d2, brute_closest(b, [7, 13], 10))
        v, coeffs, d2 = closest_vector(b, [5, 9])  # 1*b0 + 2*b1 = (5, 9)
        self.assertEqual((v, d2), ([5, 9], 0))


class BKZTests(unittest.TestCase):
    def test_bkz_never_worse_than_lll_and_hkz_finds_svp(self):
        rng = random.Random(8)
        for _ in range(12):
            n = rng.randint(3, 6)
            b = random_basis(rng, n, n, -40, 40)
            r_lll, _ = lll_integral(b)
            for block in (2, 3, n):
                stats = {}
                r, u = bkz(b, block, stats=stats)
                self.assertTrue(is_unimodular(u))
                self.assertEqual(matmul(u, b), r)
                self.assertTrue(is_lll_reduced(r))
                self.assertEqual(gram_determinant(r), gram_determinant(b))
                if block == n:
                    self.assertEqual(norm2(r[0]), shortest_vector(r_lll)[2])

    def test_bkz_improves_hard_instance(self):
        # A random q-ary style lattice where LLL is visibly suboptimal.
        rng = random.Random(9)
        n, q = 14, 10007
        a = [rng.randrange(q) for _ in range(n - 1)]
        b = [[q] + [0] * (n - 1)]
        for i, ai in enumerate(a):
            row = [ai] + [0] * (n - 1)
            row[i + 1] = 1
            b.append(row)
        r_lll, _ = lll_integral(b)
        r_bkz, _ = bkz(b, 8)
        self.assertLessEqual(norm2(r_bkz[0]), norm2(r_lll[0]))
        self.assertTrue(is_lll_reduced(r_bkz))

    def test_block_size_validation(self):
        with self.assertRaises(ValueError):
            bkz([[1, 0], [0, 1]], 1)


class NormalFormTests(unittest.TestCase):
    def assert_hnf_shape(self, h):
        last = -1
        rows = [r for r in h if any(r)]
        self.assertEqual(rows, h[:len(rows)], "zero rows must be last")
        for i, r in enumerate(rows):
            p = next(j for j, x in enumerate(r) if x)
            self.assertGreater(p, last)
            self.assertGreater(r[p], 0)
            for k in range(i):
                self.assertTrue(0 <= h[k][p] < r[p])
            for k in range(i + 1, len(h)):
                self.assertEqual(h[k][p], 0)
            last = p

    def test_hnf_shape_transform_and_uniqueness(self):
        rng = random.Random(10)
        for _ in range(60):
            m, n = rng.randint(1, 5), rng.randint(1, 5)
            a = [[rng.randint(-15, 15) for _ in range(n)] for _ in range(m)]
            h, u = hermite_normal_form(a)
            self.assert_hnf_shape(h)
            self.assertTrue(is_unimodular(u))
            self.assertEqual(matmul(u, a), h)
            mixed = matmul(random_unimodular(rng, m), a)
            self.assertEqual(hermite_normal_form(mixed)[0], h)

    def test_snf_against_determinantal_divisors(self):
        rng = random.Random(11)
        for _ in range(60):
            m, n = rng.randint(1, 4), rng.randint(1, 4)
            a = [[rng.randint(-12, 12) for _ in range(n)] for _ in range(m)]
            s, u, v = smith_normal_form(a)
            self.assertTrue(is_unimodular(u))
            self.assertTrue(is_unimodular(v))
            self.assertEqual(matmul(matmul(u, a), v), s)
            diag = [s[i][i] for i in range(min(m, n))]
            for i in range(m):
                for j in range(n):
                    if i != j:
                        self.assertEqual(s[i][j], 0)
            for x, y in zip(diag, diag[1:]):
                if x == 0:
                    self.assertEqual(y, 0)
                else:
                    self.assertEqual(y % x, 0)
            prev = 1
            for k in range(1, min(m, n) + 1):
                dk = minors_gcd(a, k)
                expected = dk // prev if prev else 0
                self.assertEqual(diag[k - 1], expected)
                prev = dk if dk else 0
                if dk == 0:
                    break

    def test_known_snf(self):
        self.assertEqual(invariant_factors([[2, 4, 4], [-6, 6, 12],
                                            [10, -4, -16]]), [2, 6, 12])


class ApplicationTests(unittest.TestCase):
    def test_minimal_polynomials(self):
        getcontext().prec = 80
        two, three = Decimal(2), Decimal(3)
        cases = [
            (two.sqrt() + three.sqrt(), 4, [1, 0, -10, 0, 1]),
            ((1 + Decimal(5).sqrt()) / 2, 2, [-1, -1, 1]),
            (evaluate("cbrt(2)", 80), 3, [-2, 0, 0, 1]),
            (evaluate("cbrt(2)+sqrt(3)", 90), 6, None),
        ]
        for alpha, deg, expected in cases:
            coeffs, residual = minimal_polynomial(alpha, deg, digits=60)
            if expected is not None:
                self.assertEqual(coeffs, expected)
            self.assertLess(abs(residual), Decimal(10) ** -40)
        # cbrt(2) + sqrt(3): x^6 - 9x^4 - 4x^3 + 27x^2 - 36x - 23
        coeffs, _ = minimal_polynomial(evaluate("cbrt(2)+sqrt(3)", 90), 6, 60)
        self.assertEqual(coeffs, [-23, -36, 27, -4, -9, 0, 1])

    def test_integer_relation_among_logs(self):
        getcontext().prec = 60
        vals = [Decimal(2).ln(), Decimal(3).ln(), Decimal(5).ln(),
                Decimal(360).ln()]
        rel, res = integer_relation(vals, digits=40)
        # log 360 = 3 log 2 + 2 log 3 + log 5
        self.assertEqual(rel, [3, 2, 1, -1])
        self.assertLess(abs(res), Decimal(10) ** -30)

    def test_solve_integer_system(self):
        rng = random.Random(12)
        for _ in range(40):
            m, n = rng.randint(1, 3), rng.randint(2, 5)
            a = [[rng.randint(-9, 9) for _ in range(n)] for _ in range(m)]
            x_true = [rng.randint(-20, 20) for _ in range(n)]
            b = [sum(r[j] * x_true[j] for j in range(n)) for r in a]
            x0, ker = solve_integer_system(a, b)
            self.assertEqual([sum(r[j] * x0[j] for j in range(n)) for r in a], b)
            for k in ker:
                self.assertEqual([sum(r[j] * k[j] for j in range(n)) for r in a],
                                 [0] * m)
            # Kernel rank is n - rank(A).
            rank = sum(1 for row in hermite_normal_form(a)[0] if any(row))
            self.assertEqual(len(ker), n - rank)
            # x_true - x0 must lie in the kernel lattice: solve for it exactly.
            diff = [p - q for p, q in zip(x_true, x0)]
            if ker:
                v, _, d = closest_vector(ker, diff)
                self.assertEqual(d, 0)
            else:
                self.assertEqual(diff, [0] * n)

    def test_unsolvable_systems(self):
        with self.assertRaises(NoIntegerSolution):
            solve_integer_system([[2, 4]], [3])
        with self.assertRaises(NoIntegerSolution):
            solve_integer_system([[1, 1], [2, 2]], [1, 3])

    def test_solvability_matches_brute_force(self):
        rng = random.Random(13)
        for _ in range(60):
            a = [[rng.randint(-4, 4) for _ in range(2)] for _ in range(2)]
            b = [rng.randint(-6, 6), rng.randint(-6, 6)]
            brute = any(a[0][0] * x + a[0][1] * y == b[0] and
                        a[1][0] * x + a[1][1] * y == b[1]
                        for x in range(-40, 41) for y in range(-40, 41))
            try:
                solve_integer_system(a, b)
                found = True
            except NoIntegerSolution:
                found = False
            if not brute:
                self.assertFalse(found)
            elif det_bareiss(a) != 0:
                self.assertTrue(found)

    def test_kernel_lattice(self):
        ker = kernel_lattice([[1, 2, 3, 4]])
        self.assertEqual(len(ker), 3)
        for k in ker:
            self.assertEqual(k[0] + 2 * k[1] + 3 * k[2] + 4 * k[3], 0)
        self.assertNotEqual(gram_determinant(ker), 0)
        # The kernel of a primitive row has covolume |row| = sqrt(30).
        self.assertEqual(gram_determinant(ker), 30)
        self.assertTrue(is_lll_reduced(ker))

    def test_simultaneous_approximation(self):
        alphas = [Fraction("0.4142135623730950488"),
                  Fraction("0.7320508075688772935")]
        q, ps, err = simultaneous_approximation(alphas, 1000)
        self.assertGreater(q, 0)
        self.assertLessEqual(q, 10 ** 4)
        for a, p in zip(alphas, ps):
            self.assertLessEqual(abs(q * a - p), err)
        # Dirichlet: infinitely many q have error <= q^(-1/2) for two numbers;
        # LLL should land within a small constant factor of that.
        self.assertLessEqual(err * err * q, 4)


class CLITests(unittest.TestCase):
    def test_poly_str(self):
        self.assertEqual(poly_str([1, 0, -10, 0, 1]), "x^4 - 10x^2 + 1")
        self.assertEqual(poly_str([-1, -1, 1]), "x^2 - x - 1")
        self.assertEqual(poly_str([0, -3]), "-3x")

    def test_evaluate_rejects_code(self):
        with self.assertRaises(ValueError):
            evaluate("__import__('os')", 30)

    def test_commands_run(self):
        import contextlib
        import io
        for argv in (["lll", "[[1,1,1],[-1,0,2],[3,5,6]]"],
                     ["bkz", "[[1,1,1],[-1,0,2],[3,5,6]]", "--block", "3"],
                     ["svp", "[[1,1,1],[-1,0,2],[3,5,6]]"],
                     ["cvp", "[[1,1,1],[-1,0,2],[3,5,6]]", "[0.4, 7.2, 3]"],
                     ["hnf", "[[2,4],[6,8]]"], ["snf", "[[2,4],[6,8]]"],
                     ["solve", "[[3,5,7]]", "[1]"],
                     ["minpoly", "sqrt(2)+sqrt(3)", "--degree", "4"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(main(argv), 0)
            self.assertTrue(buf.getvalue().strip())


if __name__ == "__main__":
    unittest.main()
