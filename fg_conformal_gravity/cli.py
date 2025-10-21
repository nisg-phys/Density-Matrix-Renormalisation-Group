from __future__ import annotations
import argparse
import sympy as sp

from .solver import solve_fg_conformal_gravity_series


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve 4D conformal gravity in FG gauge (diagonal boundary metric)")
    parser.add_argument("--series-order", type=int, default=4, help="Max power in G_ii(z) (>=2)")
    parser.add_argument("--max-bach-power", type=int, default=4, help="Max power for z-series of Bach components")
    parser.add_argument("--g0", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Boundary metric diag entries (t,x,y)")

    args = parser.parse_args()

    result = solve_fg_conformal_gravity_series(series_order=args.series_order,
                                               boundary_metric_values=tuple(args.g0),
                                               max_bach_power=args.max_bach_power)
    sol = result["solution"]

    if not sol:
        print("No nontrivial solution found under current truncation and assumptions.")
        return

    print("Solved subleading coefficients (diagonal ansatz):")
    for k in sorted(sol.keys(), key=lambda s: str(s)):
        print(f"  {k} = {sp.simplify(sol[k])}")


if __name__ == "__main__":
    main()
