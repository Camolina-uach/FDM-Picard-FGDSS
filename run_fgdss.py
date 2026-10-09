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

"""Entry point: run the full FGDSS experiment suite and generate the figures.

Usage::

    python run_fgdss.py

This reproduces the numerical results of Chapter 5: it runs the nine experiments
(``outputs/expN_.../``) and renders the figures (``outputs/figures/``). The
parameters below are the reference thesis-suite values. On the tested machine,
the complete suite takes roughly 10--20 minutes because the reference solver
uses dense direct factorizations; timing varies with the numerical libraries
and hardware (see experiment 9).
"""
from pathlib import Path

from fgdss_fd_operators import FDGrid
from fgdss_solver import FGDSSParams
from fgdss_experiments import run_all_experiments
from fgdss_plots import plot_all
from fgdss_tables import write_all_tables

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "outputs"

# Rectangular domain B = [xL, xR] x [yL, yR] with M, N subintervals.
grid = FDGrid(xL=-20.0, xR=20.0, yL=-20.0, yR=20.0, M=48, N=48)

# FGDSS coefficients and fractional (Riesz) orders; theta=None uses the
# structural default sqrt((phi-psi)(eta-chi)). See FGDSSParams for the meaning.
params = FGDSSParams(
    alpha=1.0, beta=1.0, gamma=1.0, xi=1.0,
    psi=1.0, eta=2.0, phi=2.0, chi=1.0, theta=None,
    lam=1.2, mu=1.2, nu=1.2, kappa=1.2, rho=1.2, sigma=1.2,
    dt=5.0e-4, t_final=2.0e-2,
    picard_tol=1.0e-10, picard_max_iter=30, picard_relaxation=1.0,
    fail_on_nonconvergence=True,
)

if __name__ == "__main__":
    print("Running FGDSS FDM+Picard experiments...")
    run_all_experiments(OUTPUT_ROOT, grid=grid, params=params)
    print("Generating figures...")
    plot_all(OUTPUT_ROOT)
    print("Generating LaTeX tables...")
    write_all_tables(OUTPUT_ROOT)
    print(f"Done. Outputs written to {OUTPUT_ROOT}")
