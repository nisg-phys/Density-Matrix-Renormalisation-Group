from __future__ import annotations
from typing import Dict, Tuple, List
import sympy as sp

from .fg import build_fg_metric_diagonal
from .tensors import bach_tensor


def _collect_series_equations(expr: sp.Expr, z: sp.Symbol, max_power: int) -> Dict[int, sp.Expr]:
    """Return a dict mapping power k to coeff of z**k in series of expr up to max_power.
    Negative powers are included if present.
    """
    # Expand as a series around z=0 up to z**max_power
    ser = sp.series(sp.simplify(expr), z, 0, max_power + 1).removeO()
    # Collect coefficients of each power of z in the polynomial/series
    coeffs: Dict[int, sp.Expr] = {}
    # Identify minimal and maximal exponents present
    terms = sp.Add.make_args(sp.expand(ser))
    for term in terms:
        # term is coeff * z**k (or just coeff)
        if term.has(z):
            p = sp.Poly(term, z)
            for monom, c in p.terms():
                k = monom[0]
                coeffs[k] = coeffs.get(k, 0) + c
        else:
            coeffs[0] = coeffs.get(0, 0) + term
    # Simplify coefficients
    for k in list(coeffs.keys()):
        coeffs[k] = sp.simplify(coeffs[k])
    return coeffs


def solve_fg_conformal_gravity_series(series_order: int = 4,
                                       boundary_metric_values: Tuple[float | int, float | int, float | int] = (1, 1, 1),
                                       max_bach_power: int = 4) -> Dict[str, object]:
    """Solve B_ab=0 order-by-order for diagonal FG ansatz in 4D.

    Parameters
    - series_order: highest z power used in G_ii(z) (>=2).
    - boundary_metric_values: numerical values for (g0_t, g0_x, g0_y).
    - max_bach_power: expand Bach components in z up to this power.

    Returns
    - dict with keys: 'solution', 'unknowns', 'equations_per_power', 'B_components'
    """
    data = build_fg_metric_diagonal(series_order=series_order,
                                    boundary_metric=tuple(map(sp.Integer, boundary_metric_values)))
    coords = data["coords"]
    z = data["z"]
    g = data["metric"]
    coeff_symbols = data["coeff_symbols"]

    # Unknowns: all gk_* with k>=2
    unknowns: List[sp.Symbol] = []
    for k, (gt, gx, gy) in coeff_symbols.items():
        if k == 0:
            continue
        unknowns.extend([gt, gx, gy])

    # Compute Bach tensor
    B = bach_tensor(coords, g)

    # Build equations by expanding each independent component's series and setting coefficients to zero
    components_to_use = [(0, 0), (1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]
    eqs_by_power: Dict[int, List[sp.Expr]] = {}
    B_components: Dict[Tuple[int, int], Dict[int, sp.Expr]] = {}

    for (a, b) in components_to_use:
        expr = sp.simplify(B[a, b])
        coeffs = _collect_series_equations(expr, z, max_bach_power)
        B_components[(a, b)] = coeffs
        for k, c in coeffs.items():
            if k > max_bach_power:
                continue
            eqs_by_power.setdefault(k, []).append(sp.simplify(c))

    # Flatten equations up to max_bach_power and solve
    all_eqs: List[sp.Eq] = []
    for k in sorted(eqs_by_power.keys()):
        if k <= max_bach_power:
            for c in eqs_by_power[k]:
                all_eqs.append(sp.Eq(sp.simplify(c), 0))

    sol = sp.solve(all_eqs, unknowns, dict=True)
    solution = sol[0] if sol else {}

    return {
        "solution": solution,
        "unknowns": unknowns,
        "equations_per_power": eqs_by_power,
        "B_components": B_components,
    }


if __name__ == "__main__":
    result = solve_fg_conformal_gravity_series(series_order=4,
                                               boundary_metric_values=(1, 1, 1),
                                               max_bach_power=4)
    print("Solved unknowns:")
    for sym, val in result["solution"].items():
        print(f"  {sym} = {sp.simplify(val)}")
