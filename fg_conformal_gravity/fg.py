from __future__ import annotations
from typing import Dict, Tuple
import sympy as sp


def build_fg_metric_diagonal(series_order: int = 4,
                             boundary_metric: Tuple[sp.Expr, sp.Expr, sp.Expr] | None = None,
                             coord_names: Tuple[str, str, str, str] = ("z", "t", "x", "y")) -> Dict[str, object]:
    """Construct a 4D Fefferman–Graham metric with diagonal 3D boundary block.

    The ansatz is
        ds^2 = (dz^2 + G_ij(z) dx^i dx^j) / z^2,
    with G_ij diagonal and expanded as
        G_ii(z) = g0_i + z^2 g2_i + z^3 g3_i + z^4 g4_i + ... (truncated)

    Parameters
    - series_order: truncate the G_ii(z) expansion at this maximum power of z.
    - boundary_metric: optional tuple (g0_t, g0_x, g0_y); default is Euclidean (1,1,1).
    - coord_names: names for (z, t, x, y).

    Returns
    - dict with keys: 'coords', 'z', 'metric', 'G_series', 'coeff_symbols'
    """
    if series_order < 2:
        raise ValueError("series_order must be >= 2 for nontrivial subleading terms")

    z, t, x, y = sp.symbols("%s %s %s %s" % coord_names, real=True)

    if boundary_metric is None:
        g0_t, g0_x, g0_y = sp.symbols("g0_t g0_x g0_y", real=True)
    else:
        g0_t, g0_x, g0_y = boundary_metric

    # Unknown subleading diagonal coefficients
    symbols_by_power = {0: (g0_t, g0_x, g0_y)}
    series_coeffs = {"t": {0: g0_t}, "x": {0: g0_x}, "y": {0: g0_y}}

    # Create coefficient symbols gk_i for k in [2..series_order]
    for k in range(2, series_order + 1):
        gk_t, gk_x, gk_y = sp.symbols(f"g{k}_t g{k}_x g{k}_y", real=True)
        symbols_by_power[k] = (gk_t, gk_x, gk_y)
        series_coeffs["t"][k] = gk_t
        series_coeffs["x"][k] = gk_x
        series_coeffs["y"][k] = gk_y

    # Build truncated series G_ii(z)
    G_tt = sum((z**k) * series_coeffs["t"][k] for k in series_coeffs["t"])  # type: ignore[index]
    G_xx = sum((z**k) * series_coeffs["x"][k] for k in series_coeffs["x"])  # type: ignore[index]
    G_yy = sum((z**k) * series_coeffs["y"][k] for k in series_coeffs["y"])  # type: ignore[index]

    # 4D metric components in FG gauge (diagonal boundary block)
    g = sp.MutableDenseNDimArray([[0]*4 for _ in range(4)])
    g[0, 0] = 1 / z**2  # g_zz
    g[0, 1] = g[1, 0] = 0
    g[0, 2] = g[2, 0] = 0
    g[0, 3] = g[3, 0] = 0

    g[1, 1] = G_tt / z**2
    g[2, 2] = G_xx / z**2
    g[3, 3] = G_yy / z**2

    coords = (z, t, x, y)

    return {
        "coords": coords,
        "z": z,
        "metric": g,
        "G_series": {"tt": G_tt, "xx": G_xx, "yy": G_yy},
        "coeff_symbols": symbols_by_power,
    }
