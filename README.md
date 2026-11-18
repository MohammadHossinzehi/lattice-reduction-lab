# lattice-reduction-lab

Exact lattice basis reduction in pure Python, and the useful things it unlocks: LLL (two independent implementations that agree operation for operation), BKZ with exact enumeration, shortest and closest vector search, Hermite and Smith normal forms, every integer solution of a linear system, integer relation detection and minimal polynomial recovery.

No dependencies. No NumPy. No floating point in any decision the algorithms make.

```
$ python -m lattice_lab minpoly "cbrt(2)+sqrt(3)" --degree 6
x^6 - 9x^4 - 4x^3 + 27x^2 - 36x - 23
residual: -1.000e-57
```

That polynomial was recovered from nothing but 40 decimal digits of 2.99197... LLL found the only short integer vector hiding in a 7 dimensional lattice built from those digits.

## Why lattices

A lattice is every integer combination of some basis vectors. The same lattice has infinitely many bases, most of them awful: long, nearly parallel vectors. **Lattice reduction** finds a good basis, with short and nearly orthogonal vectors. That one primitive turns out to answer a surprising range of questions:

| Question | Lattice trick |
|---|---|
| Is 2.99197... algebraic, and what is its polynomial? | short vector in the lattice of `(e_i, round(N * alpha^i))` |
| Is `log 360` an integer combination of `log 2, log 3, log 5`? | same, with the logs as the last column |
| All integer `x` with `A x = b`? | Smith normal form for one solution, LLL on the kernel for a short basis, CVP to shrink the solution |
| One denominator approximating several irrationals at once? | Lagarias' simultaneous Diophantine approximation lattice |
| How hard is a Learning With Errors instance? | how short a vector BKZ finds in a q-ary lattice |

## What is inside

```
lattice_lab/
  linalg.py         exact Gram-Schmidt, Bareiss determinant, unimodularity check
  lll.py            LLL: rational (incremental GSO in Fractions) and integral (de Weger)
  enumeration.py    exact SVP and CVP by Schnorr-Euchner enumeration, Babai nearest plane
  bkz.py            BKZ with exact block SVP and dependency free insertion
  normal_forms.py   Hermite normal form and Smith normal form, with transforms
  applications.py   integer systems, kernels, integer relations, minimal polynomials,
                    simultaneous approximation
  __main__.py       command line interface
tests/              28 oracle based tests
examples/demo.py    a guided tour with timings
```

### LLL, twice

`lll_rational` keeps `mu` and `|b*_i|^2` as Fractions and updates them in place on every size reduction and swap (Cohen, Algorithm 2.6.3). `lll_integral` is de Weger's fraction free version (Cohen 2.6.7): it only ever stores the integers `d_i = det Gram(b_1..b_i)` and `lambda_kj = d_j * mu_kj`, and every division it performs is exact.

They share a schedule, a rounding rule and an exact Lovasz test, so they must make identical decisions. The test suite runs both on 60 random bases with three different `delta` values and requires the outputs, the transformation matrices and the swap counts to be **identical**. Two implementations built on different invariants agreeing step for step is a much stronger check than either one passing a handful of examples.

The integral version is the default because it is faster: integers are cheaper than Fractions with growing denominators.

### Enumeration without epsilons

`shortest_vector` and `closest_vector` walk the Gram-Schmidt coordinates depth first, trying coefficients nearest the projected centre first so the search radius shrinks early. Floating point enumeration usually needs a fudge factor on the radius; here candidate ranges come from `math.isqrt` and every candidate is then checked with exact Fraction arithmetic, so the result is the true minimum, full stop. CVP accepts rational targets outside the span (the orthogonal part is a constant added to every distance) and starts from Babai's nearest plane answer as its initial bound.

### BKZ insertion that never creates a dependent set

Textbook BKZ inserts the new short vector into the basis and asks LLL to discard the resulting linear dependency, which needs an LLL variant that tolerates generating sets. This implementation avoids that entirely. The projected shortest vector's coefficient vector is primitive, so a short Euclid run of unimodular column operations inside the block turns it into a unit vector; the basis vector at that position *is* the new vector, and it is rotated to the front. Every operation is unimodular, so the transformation matrix stays exact and `U * input == output` is checked on every test.

### Normal forms

`hermite_normal_form` returns `(H, U)` with `U A = H`. `smith_normal_form` returns `(S, U, V)` with `U A V = S` and the divisibility chain `s_1 | s_2 | ...`. Divisibility is enforced inside the elimination loop (when the pivot does not divide a remaining entry, that row is added to the pivot row and elimination continues) rather than patched afterwards.

## Running it

Requires Python 3.9 or newer, nothing else.

```bash
git clone https://github.com/MohammadHossinzehi/lattice-reduction-lab
cd lattice-reduction-lab

python -m unittest discover -s tests -v    # full suite, about 6 seconds
python examples/demo.py                    # the guided tour
```

### Command line

Matrices are JSON or `@file` (JSON, or whitespace separated rows).

```bash
python -m lattice_lab lll  "[[1,1,1],[-1,0,2],[3,5,6]]"
python -m lattice_lab bkz  @basis.txt --block 10
python -m lattice_lab svp  "[[1,1,1],[-1,0,2],[3,5,6]]"
python -m lattice_lab cvp  "[[1,1,1],[-1,0,2],[3,5,6]]" "[0.4, 7.2, 3]"
python -m lattice_lab hnf  "[[2,4],[6,8]]"
python -m lattice_lab snf  "[[2,4,4],[-6,6,12],[10,-4,-16]]"
python -m lattice_lab solve "[[3,5,7]]" "[1]"
python -m lattice_lab minpoly "sqrt(2)+sqrt(3)" --degree 4
python -m lattice_lab relation 0.693147180559945 1.098612288668110 1.791759469228055 --digits 14
```

`minpoly` accepts decimal digits or a small expression language (`sqrt`, `cbrt`, `root(x, k)`, `pi`, `e`, arithmetic) evaluated to high precision with `decimal`. The expression is parsed with `ast` against a whitelist; it is never handed to a bare `eval`.

### Library

```python
from lattice_lab import lll, bkz, shortest_vector, closest_vector, solve_integer_system

reduced, U = lll([[1, 0, 0, 31416], [0, 1, 0, 27183], [0, 0, 1, 14142]])
v, coeffs, norm2 = shortest_vector(reduced)

x0, kernel = solve_integer_system([[12, 18, -6, 30], [5, -7, 11, 3]], [36, 4])
# every solution is x0 + an integer combination of the rows of kernel
```

## Sample output

From `examples/demo.py` on a 28 dimensional q-ary lattice (the shape used in Learning With Errors, q = 521):

```
LLL      |b1|^2 =   1212  hermite factor 1.5252  (0.08s)
BKZ-4    |b1|^2 =    999  hermite factor 1.3847  (3.87s)
BKZ-10   |b1|^2 =    999  hermite factor 1.3847  (5.64s)
exact SVP lambda_1^2 = 999  (0.51s)
```

LLL alone misses the shortest vector; even a small BKZ block finds it, which enumeration then certifies as optimal.

## Testing philosophy

Every test compares against an independent oracle, never against expected output copied from the implementation itself:

* **LLL**: both reduction conditions checked exactly, `U * B == R`, `det U = +-1`, covolume unchanged, the `2^((n-1)/2)` approximation bound checked against the true `lambda_1` from enumeration, and the rational and integral versions required to agree exactly.
* **SVP and CVP**: brute force over a box of coefficient vectors on small lattices, including rational targets outside the span.
* **BKZ**: transformation exactness, output still LLL reduced, and with block size `n` the first vector must be a shortest vector.
* **HNF**: pivot and range conditions, `U A = H`, and canonicity: mixing the rows with a random unimodular matrix must not change the result.
* **SNF**: every diagonal entry must equal `d_k / d_(k-1)`, where `d_k` is the gcd of all `k x k` minors computed by brute force.
* **Integer systems**: substitution, kernel rank equals `n - rank A`, the known planted solution must differ from the returned one by a kernel lattice vector, and solvability must agree with brute force search over a box.
* **Relations**: known minimal polynomials (including the degree 6 one above) and the relation `log 360 = 3 log 2 + 2 log 3 + log 5`.

## Limits and honest caveats

* Pure Python means this is a teaching and verification tool, not a competitor to fplll. LLL is comfortable into the low hundreds of dimensions for moderate entries; BKZ and exact enumeration into the 30s or 40s.
* `integer_relation` always returns *some* vector. If no true relation exists at the given precision you get large coefficients and a residual near `10^-digits`; that is the signal. Try `minpoly pi --degree 3` to see what a non answer looks like.
* BKZ uses full enumeration in each block without pruning, so cost grows quickly with block size.

## References

* A. K. Lenstra, H. W. Lenstra, L. Lovasz, *Factoring polynomials with rational coefficients*, 1982.
* H. Cohen, *A Course in Computational Algebraic Number Theory*, sections 2.4 to 2.7.
* C. P. Schnorr, M. Euchner, *Lattice basis reduction: improved practical algorithms*, 1994.
* L. Babai, *On Lovasz' lattice reduction and the nearest lattice point problem*, 1986.
* J. C. Lagarias, *The computational complexity of simultaneous Diophantine approximation problems*, 1985.

## License

MIT
