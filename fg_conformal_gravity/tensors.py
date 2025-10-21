from __future__ import annotations
from typing import Dict, Tuple, Iterable
import sympy as sp
import itertools

Index = Tuple[int, int]


def inverse_metric(metric: sp.MutableDenseNDimArray) -> sp.MutableDenseNDimArray:
    n = metric.shape[0]
    mat = sp.Matrix([[metric[i, j] for j in range(n)] for i in range(n)])
    inv = mat.inv()
    g_inv = sp.MutableDenseNDimArray([[0]*n for _ in range(n)])
    for i in range(n):
        for j in range(n):
            g_inv[i, j] = sp.simplify(inv[i, j])
    return g_inv


def christoffel_symbols(coords: Tuple[sp.Symbol, ...],
                        g: sp.MutableDenseNDimArray,
                        g_inv: sp.MutableDenseNDimArray) -> sp.MutableDenseNDimArray:
    n = len(coords)
    Gamma = sp.MutableDenseNDimArray([[[0]*n for _ in range(n)] for _ in range(n)])

    # Precompute partial derivatives of metric
    dg = [[ [None]*n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                dg[a][b][c] = sp.diff(g[b, c], coords[a])

    for a in range(n):
        for b in range(n):
            for c in range(n):
                # Gamma^a_{bc} = 1/2 g^{ad} (∂_b g_{dc} + ∂_c g_{db} - ∂_d g_{bc})
                val = 0
                for d in range(n):
                    val += g_inv[a, d] * (dg[b][d][c] + dg[c][d][b] - dg[d][b][c])
                Gamma[a, b, c] = sp.simplify(sp.Rational(1, 2) * val)
    return Gamma


def riemann_tensor(coords: Tuple[sp.Symbol, ...],
                   Gamma: sp.MutableDenseNDimArray) -> sp.MutableDenseNDimArray:
    n = len(coords)
    Riem = sp.MutableDenseNDimArray([[[[0]*n for _ in range(n)] for _ in range(n)] for _ in range(n)])

    # R^a_{ bcd } = ∂_c Γ^a_{bd} - ∂_d Γ^a_{bc} + Γ^a_{ce} Γ^e_{bd} - Γ^a_{de} Γ^e_{bc}
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    term = sp.diff(Gamma[a, b, d], coords[c]) - sp.diff(Gamma[a, b, c], coords[d])
                    acc = term
                    for e in range(n):
                        acc += Gamma[a, c, e] * Gamma[e, b, d] - Gamma[a, d, e] * Gamma[e, b, c]
                    Riem[a, b, c, d] = sp.simplify(acc)
    return Riem


def ricci_tensor(Riem: sp.MutableDenseNDimArray) -> sp.MutableDenseNDimArray:
    n = Riem.shape[0]
    Ric = sp.MutableDenseNDimArray([[0]*n for _ in range(n)])
    # Ric_{bd} = R^a_{ bad }
    for b in range(n):
        for d in range(n):
            acc = 0
            for a in range(n):
                acc += Riem[a, b, a, d]
            Ric[b, d] = sp.simplify(acc)
    return Ric


def ricci_scalar(g_inv: sp.MutableDenseNDimArray, Ric: sp.MutableDenseNDimArray) -> sp.Expr:
    n = Ric.shape[0]
    R = 0
    for a in range(n):
        for b in range(n):
            R += g_inv[a, b] * Ric[a, b]
    return sp.simplify(R)


def weyl_tensor(g: sp.MutableDenseNDimArray,
                g_inv: sp.MutableDenseNDimArray,
                Riem: sp.MutableDenseNDimArray,
                Ric: sp.MutableDenseNDimArray,
                R: sp.Expr) -> sp.MutableDenseNDimArray:
    n = g.shape[0]
    assert n >= 3
    C = sp.MutableDenseNDimArray([[[[0]*n for _ in range(n)] for _ in range(n)] for _ in range(n)])

    # C_{abcd} = R_{abcd} - 2/(n-2) (g_{a[c} R_{d]b} - g_{b[c} R_{d]a}) + 2 R / ((n-1)(n-2)) g_{a[c} g_{d]b}
    # First lower all indices on Riem: R_{abcd} = g_{ae} R^e_{ bcd }
    R_down = sp.MutableDenseNDimArray([[[[0]*n for _ in range(n)] for _ in range(n)] for _ in range(n)])
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    acc = 0
                    for e in range(n):
                        acc += g[a, e] * Riem[e, b, c, d]
                    R_down[a, b, c, d] = sp.simplify(acc)

    # Expand antisymmetrizations explicitly. Because g_{a[c}X_{d]b} = (1/2)(g_{ac}X_{db} - g_{ad}X_{cb}),
    # the effective multipliers become 1/(n-2) and R/((n-1)(n-2)) respectively when writing
    # the explicit differences below.
    factor1 = sp.Rational(1, 1) / (n - 2)
    factor2 = R / ((n - 1) * (n - 2))

    def antisym(ac: Tuple[int, int], val: sp.Expr) -> sp.Expr:
        # not used directly; we'll build terms manually
        return val

    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    g_ac = g[a, c]
                    g_ad = g[a, d]
                    g_bc = g[b, c]
                    g_bd = g[b, d]
                    term = R_down[a, b, c, d]
                    term -= factor1 * (g_ac * Ric[d, b] - g_ad * Ric[c, b] - g_bc * Ric[d, a] + g_bd * Ric[c, a])
                    term += factor2 * (g_ac * g[d, b] - g_ad * g[c, b])
                    C[a, b, c, d] = sp.simplify(term)
    return C


def schouten_tensor(g: sp.MutableDenseNDimArray,
                    g_inv: sp.MutableDenseNDimArray,
                    Ric: sp.MutableDenseNDimArray,
                    R: sp.Expr) -> sp.MutableDenseNDimArray:
    n = g.shape[0]
    # P_ab = 1/(n-2) * ( Ric_ab - R * g_ab / (2(n-1)) ) ; for n=4, P = 1/2 (Ric - R g / 6)
    P = sp.MutableDenseNDimArray([[0]*n for _ in range(n)])
    for a in range(n):
        for b in range(n):
            P[a, b] = sp.simplify((Ric[a, b] - (R * g[a, b]) / (2*(n-1))) / (n-2))
    return P


def covariant_derivative_covariant_tensor(coords: Tuple[sp.Symbol, ...],
                                           Gamma: sp.MutableDenseNDimArray,
                                           T: sp.MutableDenseNDimArray) -> sp.MutableDenseNDimArray:
    """Return ∇_d T_{a b ...} for a fully covariant tensor T with k indices.

    Output shape: (d, a, b, ...)
    """
    n = len(coords)
    rank = len(T.shape)
    result = sp.MutableDenseNDimArray.zeros(*((n,) + T.shape))

    # iterate over derivative index d and tensor indices i0, i1, ...
    for d in range(n):
        index_ranges = [range(s) for s in T.shape]
        for idx in itertools.product(*index_ranges):
            # partial derivative
            expr = sp.diff(T[idx], coords[d])
            # minus connections for each covariant index
            for pos, a in enumerate(idx):
                # T_{... a ...}
                corr = 0
                for s in range(n):
                    # Γ^s_{d a} T_{... s ...}
                    replaced_idx = list(idx)
                    replaced_idx[pos] = s
                    corr += Gamma[s, d, a] * T[tuple(replaced_idx)]
                expr -= corr
            result[(d,) + idx] = sp.simplify(expr)
    return result


def bach_tensor(coords: Tuple[sp.Symbol, ...],
                g: sp.MutableDenseNDimArray) -> sp.MutableDenseNDimArray:
    """Compute the Bach tensor B_ab in 4D via B_ab = ∇^c A_{abc} + P^{cd} C_{acbd}.

    A_{abc} = 2 ∇_[b P_{c]a} = ∇_b P_{ca} - ∇_c P_{ba}
    """
    n = len(coords)
    if n != 4:
        raise ValueError("This Bach implementation assumes 4D spacetime")

    g_inv = inverse_metric(g)
    Gamma = christoffel_symbols(coords, g, g_inv)
    Riem = riemann_tensor(coords, Gamma)
    Ric = ricci_tensor(Riem)
    R = ricci_scalar(g_inv, Ric)
    C = weyl_tensor(g, g_inv, Riem, Ric, R)
    P = schouten_tensor(g, g_inv, Ric, R)

    # ∇_d P_{ab}
    dP = covariant_derivative_covariant_tensor(coords, Gamma, P)

    # Define Cotton with indices A_{c a b} = ∇_a P_{b c} - ∇_b P_{a c}
    A = sp.MutableDenseNDimArray([[[0]*n for _ in range(n)] for _ in range(n)])
    for c in range(n):
        for a in range(n):
            for b in range(n):
                A[c, a, b] = sp.simplify(dP[a, b, c] - dP[b, a, c])

    # ∇_d A_{abc}
    dA = covariant_derivative_covariant_tensor(coords, Gamma, A)

    # B_ab = ∇^c A_{c a b} + P^{cd} C_{a c b d} = g^{cd} (∇_d A_{c a b}) + P^{cd} C_{a c b d}
    B = sp.MutableDenseNDimArray([[0]*n for _ in range(n)])

    # raise P^{cd}
    P_up = sp.MutableDenseNDimArray([[0]*n for _ in range(n)])
    for c in range(n):
        for d in range(n):
            acc = 0
            for r in range(n):
                for s in range(n):
                    acc += g_inv[c, r] * g_inv[d, s] * P[r, s]
            P_up[c, d] = sp.simplify(acc)

    for a in range(n):
        for b in range(n):
            term1 = 0
            for c in range(n):
                for d in range(n):
                    term1 += g_inv[c, d] * dA[d, c, a, b]

            term2 = 0
            for c in range(n):
                for d in range(n):
                    term2 += P_up[c, d] * C[a, c, b, d]

            B[a, b] = sp.simplify(term1 + term2)
    return B
