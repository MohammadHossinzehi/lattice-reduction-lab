"""Hermite and Smith normal forms over the integers, with transforms.

``hermite_normal_form(A)`` returns ``(H, U)`` with U unimodular and U A = H,
where H is in row echelon form, every pivot is positive and every entry above
a pivot lies in [0, pivot). The HNF is unique for the row lattice of A, which
makes it a canonical name for a lattice: two integer bases span the same
lattice exactly when their HNFs agree.

``smith_normal_form(A)`` returns ``(S, U, V)`` with U, V unimodular and
U A V = S diagonal, s_1 | s_2 | ... | s_r. The diagonal is the list of
invariant factors of the abelian group Z^n / (row lattice of A).
"""

from .linalg import as_int_matrix, identity


def _row_axpy(m, dst, q, src):
    if q:
        m[dst] = [a - q * b for a, b in zip(m[dst], m[src])]


def hermite_normal_form(a):
    h = as_int_matrix(a)
    m = len(h)
    n = len(h[0]) if m else 0
    u = identity(m)
    row = 0
    pivots = []
    for col in range(n):
        if row >= m:
            break
        while True:
            nz = [r for r in range(row, m) if h[r][col]]
            if not nz:
                break
            piv = min(nz, key=lambda r: abs(h[r][col]))
            h[row], h[piv] = h[piv], h[row]
            u[row], u[piv] = u[piv], u[row]
            done = True
            for r in range(row + 1, m):
                if h[r][col]:
                    q = h[r][col] // h[row][col]
                    _row_axpy(h, r, q, row)
                    _row_axpy(u, r, q, row)
                    if h[r][col]:
                        done = False
            if done:
                break
        if not h[row][col]:
            continue
        if h[row][col] < 0:
            h[row] = [-x for x in h[row]]
            u[row] = [-x for x in u[row]]
        for r in range(row):
            q = h[r][col] // h[row][col]
            _row_axpy(h, r, q, row)
            _row_axpy(u, r, q, row)
        pivots.append(col)
        row += 1
    return h, u


def hnf_rank(a):
    h, _ = hermite_normal_form(a)
    return sum(1 for r in h if any(r))


def smith_normal_form(a):
    s = as_int_matrix(a)
    m = len(s)
    n = len(s[0]) if m else 0
    u = identity(m)
    v = identity(n)

    def col_axpy(dst, q, src):
        if q:
            for mat in (s, v):
                for row in mat:
                    row[dst] -= q * row[src]

    def swap_cols(i, j):
        for mat in (s, v):
            for row in mat:
                row[i], row[j] = row[j], row[i]

    for t in range(min(m, n)):
        while True:
            nz = [(abs(s[i][j]), i, j) for i in range(t, m)
                  for j in range(t, n) if s[i][j]]
            if not nz:
                break
            _, pi, pj = min(nz)
            s[t], s[pi] = s[pi], s[t]
            u[t], u[pi] = u[pi], u[t]
            swap_cols(t, pj)
            clean = True
            for i in range(t + 1, m):
                q = s[i][t] // s[t][t]
                _row_axpy(s, i, q, t)
                _row_axpy(u, i, q, t)
                clean &= s[i][t] == 0
            for j in range(t + 1, n):
                q = s[t][j] // s[t][t]
                col_axpy(j, q, t)
                clean &= s[t][j] == 0
            if not clean:
                continue
            # Row and column are cleared; enforce divisibility on the rest.
            bad = next((i for i in range(t + 1, m)
                        if any(s[i][j] % s[t][t] for j in range(t + 1, n))),
                       None)
            if bad is None:
                break
            _row_axpy(s, t, -1, bad)  # row_t += row_bad, then loop again
            _row_axpy(u, t, -1, bad)
        if t < m and t < n and s[t][t] < 0:
            s[t] = [-x for x in s[t]]
            u[t] = [-x for x in u[t]]
    return s, u, v


def invariant_factors(a):
    s, _, _ = smith_normal_form(a)
    return [s[i][i] for i in range(min(len(s), len(s[0]) if s else 0))
            if s[i][i]]
