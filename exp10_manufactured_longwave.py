# -----------------------------------------------------------------------------
# Theoretical and Numerical Analysis of the Fractional Davey-Stewartson System
#
# PhD thesis, Doctorado en Ciencia y Tecnología (Applied Mathematics),
# Centro Universitario de los Lagos, Universidad de Guadalajara, 2026.
#
# Author:   Carlos Alberto Molina Holguín
# Advisors: Dr. Luis Armando Gallegos Infante
#           Dr. Jorge Eduardo Macías Díaz
#
# Reference implementation, released under the MIT License (see LICENSE).
# -----------------------------------------------------------------------------

"""Experiment 10 (thesis v4, Section 5.6): manufactured test of the long-wave
recovery with the mixed coupling theta active, plus a theta = 0 sensitivity check.

Smooth, localized, separable fields are prescribed on [-L, L]^2,

    w*(x, y) = B p(x/a) e(y/a),     v*(x, y) = B e(x/a) p(y/a),
    e(u) = exp(-u^2),               p(u) = u exp(-u^2),

and the sources of the forced long-wave subsystem

    psi D_x^kappa w* + eta D_y^kappa w* + theta D_xy^sigma v* = f_w,
    phi D_x^rho   v* + chi D_y^rho   v* + theta D_xy^sigma w* = f_v,

together with the coupling Q* = D_x^{mu/2} w* + D_y^{nu/2} v*, are evaluated in
closed form. The Riesz derivatives (symbol -|k|^s) of e and p are

    D^s e(u) = -(2^s Gamma((s+1)/2)/sqrt(pi)) 1F1((s+1)/2; 1/2; -u^2),
    D^s p(u) = -(2^{s+1} Gamma((s+3)/2)/sqrt(pi)) u 1F1((s+3)/2; 3/2; -u^2),

with D^s[f(x/a)] = a^{-s} (D^s f)(x/a); for separable fields the mixed operator
D_xy^sigma = D_x^{sigma/2} D_y^{sigma/2} factorizes by direction. The formulas are
checked against numerical quadrature in ``tests/test_longwave_v4.py``.

The discrete system M_h (W, V) = (f_w, f_v) is solved with the long-wave block
matrix and factorization of ``FGDSSSolver``, and (W, V, Q_h) is compared with
(w*, v*, Q*). The sensitivity check solves with theta = 0 in M_h while keeping
the sources computed with theta = 1.

Usage::

    python exp10_manufactured_longwave.py [OUTPUT_ROOT]   # default: outputs

Writes ``OUTPUT_ROOT/exp10_manufactured_longwave/manufactured_longwave.csv`` and
``OUTPUT_ROOT/tables/exp10_manufactured_longwave_table.tex``.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

from fgdss_fd_operators import FDGrid, mat, vec
from fgdss_solver import FGDSSParams, FGDSSSolver

HERE = Path(__file__).resolve().parent

# Configuration (Section 5.6)
L = 12.0          # half-width of the domain [-L, L]^2
A_WIDTH = 2.0     # width a of the Gaussian profiles
B_AMP = 1.0       # amplitude B
GRIDS = (16, 24, 32, 48, 64)
ORDERS = (1.2, 1.5, 1.8, 2.0)   # common value of all fractional orders
COEFS = dict(psi=1.0, eta=2.0, phi=2.0, chi=1.0, theta=1.0)  # as in Chapter 5


def riesz_e(s: float, u: float) -> float:
    """Riesz derivative of order s of e(u) = exp(-u^2)."""
    return float(-(2**s) * mp.gamma((s + 1) / 2) / mp.sqrt(mp.pi)
                 * mp.hyp1f1((s + 1) / 2, 0.5, -u * u))


def riesz_p(s: float, u: float) -> float:
    """Riesz derivative of order s of p(u) = u exp(-u^2)."""
    return float(-(2 ** (s + 1)) * mp.gamma((s + 3) / 2) / mp.sqrt(mp.pi)
                 * u * mp.hyp1f1((s + 3) / 2, 1.5, -u * u))


def profiles(x: np.ndarray, s: float):
    """1D profiles e, p and their Riesz derivatives of order s (scaled by A_WIDTH)."""
    u = x / A_WIDTH
    e = np.exp(-u * u)
    p = u * e
    De = np.array([riesz_e(s, ui) for ui in u]) * A_WIDTH ** (-s)
    Dp = np.array([riesz_p(s, ui) for ui in u]) * A_WIDTH ** (-s)
    return e, p, De, Dp


def manufactured(grid: FDGrid, s: float):
    """Exact fields, sources, and coupling at the interior nodes (common order s)."""
    x, y = grid.x, grid.y
    ex, px, Dex, Dpx = profiles(x, s)            # full order (kappa = rho = s)
    ey, py, Dey, Dpy = profiles(y, s)
    _, _, Dex_h, Dpx_h = profiles(x, s / 2)      # half order (sigma/2, mu/2, nu/2)
    _, _, Dey_h, Dpy_h = profiles(y, s / 2)
    c, B = COEFS, B_AMP
    w = B * np.outer(px, ey)
    v = B * np.outer(ex, py)
    diag_w = B * (c["psi"] * np.outer(Dpx, ey) + c["eta"] * np.outer(px, Dey))
    diag_v = B * (c["phi"] * np.outer(Dex, py) + c["chi"] * np.outer(ex, Dpy))
    mixed_v = B * np.outer(Dex_h, Dpy_h)         # D_xy^s v* = D_x^{s/2} D_y^{s/2} v*
    mixed_w = B * np.outer(Dpx_h, Dey_h)         # D_xy^s w*
    fw = diag_w + c["theta"] * mixed_v
    fv = diag_v + c["theta"] * mixed_w
    Q = B * (np.outer(Dpx_h, ey) + np.outer(ex, Dpy_h))
    ratio = math.sqrt((np.sum((c["theta"] * mixed_v) ** 2) + np.sum((c["theta"] * mixed_w) ** 2))
                      / (np.sum(diag_w**2) + np.sum(diag_v**2)))
    return w, v, fw, fv, Q, ratio


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """Relative error ``||a - b|| / ||b||`` in the Frobenius norm."""
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def recover(grid: FDGrid, s: float, theta: float, fw: np.ndarray, fv: np.ndarray):
    """Solve M_h (W, V) = (f_w, f_v) with the reference solver's long-wave matrix."""
    params = FGDSSParams(lam=s, mu=s, nu=s, kappa=s, rho=s, sigma=s,
                         psi=COEFS["psi"], eta=COEFS["eta"], phi=COEFS["phi"],
                         chi=COEFS["chi"], theta=theta)
    solver = FGDSSSolver(grid, params)
    sol = solver._long_lu.solve(np.concatenate([vec(fw), vec(fv)]))
    n = grid.ntot
    W, V = mat(sol[:n], grid), mat(sol[n:], grid)
    Qh = solver.ops.apply_x(s / 2, W) + solver.ops.apply_y(s / 2, V)
    return W, V, Qh


def run_case(s: float, M: int) -> dict:
    """One manufactured-solution case for order ``s`` on an ``M x M`` grid.

    Returns the relative errors of ``W, V, Q_h`` with ``theta`` in the solver
    and with ``theta = 0``, the ratio of mixed to diagonal source terms, and
    the largest exact value on the outermost interior ring.
    """
    grid = FDGrid(xL=-L, xR=L, yL=-L, yR=L, M=M, N=M)
    w, v, fw, fv, Q, ratio = manufactured(grid, s)
    W, V, Qh = recover(grid, s, COEFS["theta"], fw, fv)
    W0, V0, Q0 = recover(grid, s, 0.0, fw, fv)
    ring = max(np.abs(w[[0, -1], :]).max(), np.abs(w[:, [0, -1]]).max())
    return dict(s=s, M=M, h=grid.hx, err_W=rel(W, w), err_V=rel(V, v), err_Q=rel(Qh, Q),
                err_W_theta0=rel(W0, w), err_V_theta0=rel(V0, v), err_Q_theta0=rel(Q0, Q),
                coupling_ratio=ratio, outer_ring_max=float(ring))


def run(output_root: Path) -> list[dict]:
    """Run every order and grid, print the errors, and write the CSV file and
    the LaTeX table of Section 5.6 under ``output_root``.
    """
    rows = []
    for s in ORDERS:
        prev = None
        for M in GRIDS:
            row = run_case(s, M)
            if prev is not None:
                for k in ("err_W", "err_V", "err_Q"):
                    row["ord_" + k[4:]] = (math.log(prev[k] / row[k])
                                           / math.log(prev["h"] / row["h"]))
            rows.append(row)
            prev = row
            print(f"s={s:3.1f} M={M:3d} h={row['h']:6.3f}  W {row['err_W']:.3e}"
                  f"  Q {row['err_Q']:.3e}   theta=0: W {row['err_W_theta0']:.3e}"
                  f"  Q {row['err_Q_theta0']:.3e}   mixed/diag {row['coupling_ratio']:.3f}")
    outdir = output_root / "exp10_manufactured_longwave"
    outdir.mkdir(parents=True, exist_ok=True)
    keys = ["s", "M", "h", "err_W", "ord_W", "err_V", "ord_V", "err_Q", "ord_Q",
            "err_W_theta0", "err_V_theta0", "err_Q_theta0", "coupling_ratio", "outer_ring_max"]
    with (outdir / "manufactured_longwave.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in keys})
    write_table(rows, output_root / "tables")
    return rows


def _sci(x: float) -> str:
    """Format a number as ``m.mm x 10^e`` in LaTeX math mode."""
    m, e = f"{x:.2e}".split("e")
    return f"\\({m}\\times10^{{{int(e)}}}\\)"


def write_table(rows: list[dict], tabdir: Path) -> None:
    """LaTeX table for the reference order 1.2 (err_V = err_W by the x<->y symmetry)."""
    ref = [r for r in rows if r["s"] == 1.2]
    lines = [r"\begin{tabular}{llllllll}", r"\toprule",
             r" & & \multicolumn{4}{c}{\(\theta=1\) in the solver} & \multicolumn{2}{c}{\(\theta=0\) in the solver} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
             r"\(M=N\) & \(h\) & \(\mathrm{err}_W\) & order & \(\mathrm{err}_{\mathcal{Q}}\) & order & \(\mathrm{err}_W\) & \(\mathrm{err}_{\mathcal{Q}}\) \\",
             r"\midrule"]
    for r in ref:
        oW = f"\\({r['ord_W']:.2f}\\)" if "ord_W" in r else "---"
        oQ = f"\\({r['ord_Q']:.2f}\\)" if "ord_Q" in r else "---"
        lines.append(f"\\({r['M']}\\) & \\({r['h']:.3f}\\) & {_sci(r['err_W'])} & {oW} & "
                     f"{_sci(r['err_Q'])} & {oQ} & {_sci(r['err_W_theta0'])} & {_sci(r['err_Q_theta0'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tabdir.mkdir(parents=True, exist_ok=True)
    (tabdir / "exp10_manufactured_longwave_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "outputs"
    if not root.is_absolute():
        root = HERE / root
    run(root)
