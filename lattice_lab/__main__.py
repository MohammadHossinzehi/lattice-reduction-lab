"""Command line interface: ``python -m lattice_lab <command> ...``.

Matrices are given as JSON (``"[[1,2],[3,4]]"``) or as ``@path`` to a file
holding JSON or whitespace separated rows, one row per line.
"""

import argparse
import json
import sys
from decimal import Decimal, getcontext
from fractions import Fraction

from . import (bkz, closest_vector, hermite_normal_form, integer_relation,
               lll, minimal_polynomial, shortest_vector, smith_normal_form,
               solve_integer_system)
from .applications import NoIntegerSolution


def load_matrix(spec):
    text = spec
    if spec.startswith("@"):
        with open(spec[1:], encoding="utf-8") as fh:
            text = fh.read()
    text = text.strip()
    if text.startswith("["):
        return json.loads(text)
    return [[int(x) for x in line.split()] for line in text.splitlines()
            if line.strip()]


def load_vector(spec):
    text = spec.strip()
    if text.startswith("["):
        return [Fraction(str(x)) for x in json.loads(text)]
    return [Fraction(x) for x in text.replace(",", " ").split()]


def show(rows):
    width = max((len(str(x)) for r in rows for x in r), default=1)
    for r in rows:
        print("[" + " ".join(str(x).rjust(width) for x in r) + "]")


def poly_str(coeffs):
    terms = []
    for power in range(len(coeffs) - 1, -1, -1):
        c = coeffs[power]
        if c == 0:
            continue
        sign = "-" if c < 0 else "+"
        mag = abs(c)
        body = {0: f"{mag}", 1: "x" if mag == 1 else f"{mag}x"}.get(
            power, f"x^{power}" if mag == 1 else f"{mag}x^{power}")
        terms.append((sign, body))
    if not terms:
        return "0"
    first = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    return " ".join([first] + [f"{s} {b}" for s, b in terms[1:]])


def main(argv=None):
    p = argparse.ArgumentParser(prog="lattice_lab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("lll", help="LLL reduce a basis (rows)")
    s.add_argument("matrix")
    s.add_argument("--delta", default="99/100")
    s.add_argument("--method", choices=["integral", "rational"], default="integral")

    s = sub.add_parser("bkz", help="BKZ reduce a basis")
    s.add_argument("matrix")
    s.add_argument("--block", type=int, default=10)

    s = sub.add_parser("svp", help="exact shortest nonzero vector")
    s.add_argument("matrix")

    s = sub.add_parser("cvp", help="exact closest lattice vector to a target")
    s.add_argument("matrix")
    s.add_argument("target")

    s = sub.add_parser("hnf", help="Hermite normal form")
    s.add_argument("matrix")

    s = sub.add_parser("snf", help="Smith normal form / invariant factors")
    s.add_argument("matrix")

    s = sub.add_parser("solve", help="all integer solutions of A x = b")
    s.add_argument("matrix")
    s.add_argument("rhs")

    s = sub.add_parser("relation", help="integer relation among real numbers")
    s.add_argument("values", nargs="+")
    s.add_argument("--digits", type=int, default=30)

    s = sub.add_parser("minpoly", help="minimal polynomial from a decimal")
    s.add_argument("alpha", help="decimal digits, or an expression such as "
                   "'sqrt(2)+sqrt(3)' using sqrt/cbrt/root(x,k)")
    s.add_argument("--degree", type=int, required=True)
    s.add_argument("--digits", type=int, default=40)

    a = p.parse_args(argv)

    if a.cmd == "lll":
        red, _ = lll(load_matrix(a.matrix), Fraction(a.delta), a.method)
        show(red)
    elif a.cmd == "bkz":
        red, _ = bkz(load_matrix(a.matrix), a.block)
        show(red)
    elif a.cmd == "svp":
        basis, _ = lll(load_matrix(a.matrix))
        v, _, n2 = shortest_vector(basis)
        print("shortest vector:", v)
        print("squared norm:   ", n2)
    elif a.cmd == "cvp":
        basis, _ = lll(load_matrix(a.matrix))
        v, _, d2 = closest_vector(basis, load_vector(a.target))
        print("closest vector:  ", v)
        print("squared distance:", d2)
    elif a.cmd == "hnf":
        h, _ = hermite_normal_form(load_matrix(a.matrix))
        show(h)
    elif a.cmd == "snf":
        sm, _, _ = smith_normal_form(load_matrix(a.matrix))
        show(sm)
    elif a.cmd == "solve":
        try:
            x0, ker = solve_integer_system(load_matrix(a.matrix),
                                           [int(x) for x in load_vector(a.rhs)])
        except NoIntegerSolution as exc:
            print("no integer solution:", exc)
            return 1
        print("particular solution:", x0)
        print("kernel basis:")
        show(ker) if ker else print("  (trivial, the solution is unique)")
    elif a.cmd == "relation":
        rel, res = integer_relation(a.values, a.digits)
        print("relation:", rel)
        print("residual:", f"{res:.3e}")
    elif a.cmd == "minpoly":
        alpha = evaluate(a.alpha, a.digits + 20)
        coeffs, res = minimal_polynomial(alpha, a.degree, a.digits)
        print(poly_str(coeffs))
        print("residual:", f"{res:.3e}")
    return 0


def evaluate(expr, prec):
    """Evaluate a small arithmetic expression to ``prec`` digits with Decimal."""
    getcontext().prec = prec
    if all(ch in "0123456789.-+eE" for ch in expr):
        return Decimal(expr)

    def root(x, k):
        x = Decimal(x)
        if x < 0 and k % 2:
            return -root(-x, k)
        return (x.ln() / k).exp()

    env = {"sqrt": lambda x: Decimal(x).sqrt(), "cbrt": lambda x: root(x, 3),
           "root": root, "D": Decimal, "pi": _pi(prec), "e": Decimal(1).exp()}
    import ast
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                                 ast.Call, ast.Name, ast.Load, ast.Constant,
                                 ast.operator, ast.unaryop)):
            raise ValueError(f"unsupported syntax in {expr!r}")
        if isinstance(node, ast.Name) and node.id not in env:
            raise ValueError(f"unknown name {node.id!r}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("only numeric constants are allowed")
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise ValueError("only plain function calls are allowed")

    class Wrap(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)):
                call = ast.Call(ast.Name("D", ast.Load()),
                                [ast.Constant(str(node.value))], [])
                return ast.copy_location(call, node)
            return node

    tree = ast.fix_missing_locations(Wrap().visit(tree))
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, env)


def _pi(prec):
    getcontext().prec = prec + 5
    # Machin: pi = 16 atan(1/5) - 4 atan(1/239)
    def atan_inv(n):
        x = Decimal(1) / n
        x2 = x * x
        total, term, k = Decimal(0), x, 1
        eps = Decimal(10) ** -(prec + 5)
        while term > eps:
            total += term / k if (k // 2) % 2 == 0 else -term / k
            term *= x2
            k += 2
        return total
    return 16 * atan_inv(5) - 4 * atan_inv(239)


if __name__ == "__main__":
    sys.exit(main())
