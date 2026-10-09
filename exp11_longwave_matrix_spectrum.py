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

"""Experiment 11 (thesis v4, Section 5.1): smallest eigenvalue of -M_h.

Checks numerically hypothesis (ii) of the Assumption "Local solvability setting"
of Chapter 4: the symmetric long-wave block matrix

    M_h = [[psi D_x^kappa + eta D_y^kappa,  theta D_xy^sigma        ],
           [theta D_xy^sigma,               phi D_x^rho + chi D_y^rho]]

assembled by ``FGDSSSolver`` must be invertible on the Dirichlet grid. Since M_h
is symmetric, it suffices that -M_h is positive definite; its smallest
eigenvalue is computed by shift-invert Lanczos (``eigsh`` with sigma = 0) for
every grid, domain, and common order used in Chapter 5. The unit tests compare
``eigsh`` with a dense ``eigvalsh`` on small grids.

Usage::

    python exp11_longwave_matrix_spectrum.py [OUTPUT_ROOT]   # default: outputs

Writes ``OUTPUT_ROOT/exp11_longwave_matrix_spectrum/longwave_matrix_spectrum.csv``.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from scipy.sparse.linalg import eigsh

from fgdss_fd_operators import FDGrid
from fgdss_solver import FGDSSParams, FGDSSSolver

HERE = Path(__file__).resolve().parent


def min_eig_minus_M(L: float, M: int, s: float):
    """Smallest eigenvalue of -M_h on [-L, L]^2 with M = N subintervals, orders s."""
    grid = FDGrid(xL=-L, xR=L, yL=-L, yR=L, M=M, N=M)
    params = FGDSSParams(lam=s, mu=s, nu=s, kappa=s, rho=s, sigma=s)
    A = -FGDSSSolver(grid, params).Mlong.tocsc()
    asym = abs(A - A.T).max()
    lam0 = eigsh(A, k=1, sigma=0.0, which="LM", return_eigenvectors=False)[0]
    return float(lam0), float(asym), grid.hx


def cases():
    """All ``(label, L, M, s)`` combinations of domain, grid, and common order
    used in Chapter 5.
    """
    out = []
    # Refinement, conservation, fractional sweep, momentum and cost runs: [-20,20]^2.
    for s in (1.2, 1.5, 1.8, 2.0):
        for M in (16, 24, 32, 48, 64):
            out.append(("[-20,20]^2", 20.0, M, s))
    # Controlled domain sweep at h = 1.
    for L in (8, 12, 16, 24):
        out.append((f"[-{L},{L}]^2, h=1", float(L), 2 * L, 1.2))
    # Exact standing wave at integer order.
    for M in (16, 32, 64):
        out.append(("[-10,10]^2", 10.0, M, 2.0))
    return out


def run(output_root: Path) -> list[dict]:
    """Compute the smallest eigenvalue of ``-M_h`` for every case, print it, and
    write ``longwave_matrix_spectrum.csv`` under ``output_root``.
    """
    rows = []
    print("Coefficients (psi,eta,phi,chi,theta) = (1,2,2,1,1); all orders = s")
    for name, L, M, s in cases():
        lam0, asym, h = min_eig_minus_M(L, M, s)
        rows.append(dict(domain=name, M=M, h=h, s=s, lambda_min=lam0, asymmetry=asym))
        print(f"{name:>18} M={M:3d} h={h:7.4f} s={s:3.1f}  lambda_min(-M_h)={lam0:.6e}  asym={asym:.1e}")
    print(f"minimum over all cases: {min(r['lambda_min'] for r in rows):.6e}")
    outdir = output_root / "exp11_longwave_matrix_spectrum"
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "longwave_matrix_spectrum.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return rows


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "outputs"
    if not root.is_absolute():
        root = HERE / root
    run(root)
