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

"""Tests for the thesis-v4 additions (exp10, exp11, and exp12).

* closed-form Riesz derivatives of e(u) = exp(-u^2) and p(u) = u exp(-u^2)
  against numerical quadrature of the inverse Fourier transform, and against
  the classical second derivatives at order 2;
* the shift-invert eigenvalue of -M_h against a dense eigenvalue solver;
* regeneration of selected exp10 and exp11 rows against the archived reviewed
  snapshot in ``outputs_revision`` (separate tolerances for discretization-level
  quantities and for round-off-level quantities);
* exp12: the pulse statistics against the archived snapshots, the invariant
  defects, the s = 1.2 run against the archived exp2 run, and a regenerated
  short run at s = 2.
"""
from __future__ import annotations

import csv
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import exp10_manufactured_longwave as exp10  # noqa: E402
import exp11_longwave_matrix_spectrum as exp11  # noqa: E402
import exp12_fractional_order_profiles as exp12  # noqa: E402
from fgdss_fd_operators import FDGrid  # noqa: E402
from fgdss_solver import FGDSSParams, FGDSSSolver, gaussian_initial_data  # noqa: E402

SNAPSHOT = ROOT / "outputs_revision"


def _riesz_e_quad(s: float, u: float) -> float:
    """Riesz derivative of order ``s`` of ``exp(-u^2)`` at ``u``, by quadrature
    of the inverse Fourier transform.
    """
    f = lambda k: k**s * math.exp(-k * k / 4)
    val = quad(f, 0, 60)[0] if u == 0 else quad(f, 0, 60, weight="cos", wvar=u, limit=400)[0]
    return -val / math.sqrt(math.pi)


def _riesz_p_quad(s: float, u: float) -> float:
    """Riesz derivative of order ``s`` of ``u exp(-u^2)`` at ``u``, by quadrature
    of the inverse Fourier transform.
    """
    if u == 0:
        return 0.0
    f = lambda k: k ** (s + 1) * math.exp(-k * k / 4)
    return -quad(f, 0, 60, weight="sin", wvar=u, limit=400)[0] / (2 * math.sqrt(math.pi))


def _read_csv(path: Path) -> list[dict]:
    """Read a CSV file into a list of dictionaries."""
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


class ClosedFormRieszTests(unittest.TestCase):
    """Closed-form Riesz derivatives used by experiment 10."""
    def test_closed_forms_match_quadrature(self) -> None:
        """The 1F1 closed forms agree with numerical quadrature."""
        for s in (0.6, 0.9, 1.0, 1.2, 1.5, 1.8, 2.0):
            for u in (0.0, 0.3, 1.0, 2.2, 4.0):
                self.assertAlmostEqual(exp10.riesz_e(s, u), _riesz_e_quad(s, u), delta=1e-9)
                self.assertAlmostEqual(exp10.riesz_p(s, u), _riesz_p_quad(s, u), delta=1e-9)

    def test_order_two_recovers_second_derivatives(self) -> None:
        """At order 2 the closed forms reduce to minus the classical second derivative."""
        for u in (0.0, 0.7, 1.3, 2.5):
            e = math.exp(-u * u)
            self.assertAlmostEqual(exp10.riesz_e(2.0, u), (4 * u * u - 2) * e, delta=1e-13)
            self.assertAlmostEqual(exp10.riesz_p(2.0, u), (4 * u**3 - 6 * u) * e, delta=1e-13)


class LongWaveSpectrumTests(unittest.TestCase):
    """Smallest eigenvalue of -M_h (experiment 11)."""
    def test_eigsh_matches_dense_solver(self) -> None:
        """Shift-invert Lanczos agrees with a dense eigenvalue solver."""
        for L, M, s in ((20.0, 16, 1.2), (8.0, 16, 1.2), (20.0, 16, 2.0)):
            grid = FDGrid(xL=-L, xR=L, yL=-L, yR=L, M=M, N=M)
            params = FGDSSParams(lam=s, mu=s, nu=s, kappa=s, rho=s, sigma=s)
            dense_min = np.linalg.eigvalsh(-FGDSSSolver(grid, params).Mlong.toarray()).min()
            lam0, asym, _ = exp11.min_eig_minus_M(L, M, s)
            self.assertEqual(asym, 0.0)
            self.assertAlmostEqual(lam0, dense_min, delta=1e-12)
            self.assertGreater(lam0, 0.0)

    def test_regenerated_spectrum_matches_snapshot(self) -> None:
        """Recomputed eigenvalues match the archived values."""
        archived = _read_csv(SNAPSHOT / "exp11_longwave_matrix_spectrum" / "longwave_matrix_spectrum.csv")
        for row in (archived[0], archived[1], archived[-3]):
            M = int(row["M"])
            L = float(row["h"]) * M / 2.0      # domain [-L, L]^2 with M subintervals
            lam0, _, _ = exp11.min_eig_minus_M(L, M, float(row["s"]))
            self.assertAlmostEqual(lam0, float(row["lambda_min"]), delta=1e-10 * float(row["lambda_min"]) + 1e-14)


class ManufacturedLongWaveTests(unittest.TestCase):
    """Manufactured recovery of the long-wave fields (experiment 10)."""
    def test_second_order_recovery_and_theta_sensitivity(self) -> None:
        """The recovery converges with order 2, and removing theta makes the error stagnate."""
        coarse = exp10.run_case(1.2, 24)
        fine = exp10.run_case(1.2, 48)
        order_W = math.log(coarse["err_W"] / fine["err_W"]) / math.log(coarse["h"] / fine["h"])
        order_Q = math.log(coarse["err_Q"] / fine["err_Q"]) / math.log(coarse["h"] / fine["h"])
        self.assertGreater(order_W, 1.9)
        self.assertGreater(order_Q, 1.9)
        # With theta = 0 in the solver the error must not decrease.
        self.assertGreater(fine["err_W_theta0"], 10 * fine["err_W"])
        self.assertGreater(fine["err_W_theta0"], 0.9 * coarse["err_W_theta0"])

    def test_regenerated_rows_match_snapshot(self) -> None:
        """Recomputed rows match the archived CSV file."""
        archived = {(float(r["s"]), int(r["M"])): r for r in
                    _read_csv(SNAPSHOT / "exp10_manufactured_longwave" / "manufactured_longwave.csv")}
        for key in ((1.2, 16), (1.2, 32), (2.0, 24)):
            row = exp10.run_case(*key)
            ref = archived[key]
            for k in ("err_W", "err_V", "err_Q", "err_W_theta0", "err_Q_theta0", "coupling_ratio"):
                self.assertAlmostEqual(row[k], float(ref[k]), delta=1e-9 * abs(float(ref[k])))
            # Near-zero quantity: only its magnitude is compared (at most 6e-12 on
            # the coarsest grid, far below the discretization errors).
            self.assertLess(row["outer_ring_max"], 1e-11)


class FractionalOrderProfileTests(unittest.TestCase):
    """Comparison of common fractional orders (experiment 12)."""
    BASE = SNAPSHOT / "exp12_fractional_order_profiles"

    def _snapshot(self, s: float, t: float):
        """Load the archived snapshot of order ``s`` at time ``t``."""
        return np.load(self.BASE / f"s_{s:.1f}" / "snapshots" / f"t_{t:.6f}.npz")

    def test_statistics_match_snapshots(self) -> None:
        """The pulse statistics CSV agrees with the archived snapshots."""
        for row in _read_csv(self.BASE / "pulse_statistics.csv"):
            stats = exp12.pulse_stats(self._snapshot(float(row["s"]), float(row["t"])))
            for k in ("peak", "xc", "yc", "rms_radius"):
                self.assertAlmostEqual(stats[k], float(row[k]), delta=1e-12)

    def test_invariants_and_reproduction_of_exp2(self) -> None:
        """Mass and energy are conserved for every order, and s = 1.2 reproduces exp2."""
        for s in exp12.ORDERS:
            history = _read_csv(self.BASE / f"s_{s:.1f}" / "diagnostics.csv")
            self.assertLess(max(float(h["RE_M"]) for h in history), 1e-12)
            self.assertLess(max(float(h["RE_E_theta"]) for h in history), 1e-12)
        archived = np.load(SNAPSHOT / "exp2_source_free_conservation" / "final_fields.npz")
        rerun = np.load(self.BASE / "s_1.2" / "final_fields.npz")
        self.assertLess(float(np.abs(archived["U"] - rerun["U"]).max()), 1e-12)

    def test_centroid_trend_with_order(self) -> None:
        """Smaller orders move the packet less in x and more in y."""
        # Thesis Section 5.7: smaller s moves the packet less in x and more in y.
        final = [exp12.pulse_stats(self._snapshot(s, 1.0)) for s in exp12.ORDERS]
        xc = [f["xc"] for f in final]
        yc = [f["yc"] for f in final]
        self.assertEqual(xc, sorted(xc, reverse=True))
        self.assertEqual(yc, sorted(yc))

    def test_regenerated_short_run_matches_snapshot(self) -> None:
        """The first 40 steps at s = 2 reproduce the archived snapshot."""
        # First 0.1 time units at s = 2 (40 steps).
        grid = FDGrid(M=48, N=48)
        params = FGDSSParams(dt=2.5e-3, t_final=0.1, lam=2.0, mu=2.0, nu=2.0,
                             kappa=2.0, rho=2.0, sigma=2.0)
        solver = FGDSSSolver(grid, params)
        U0 = gaussian_initial_data(grid, **exp12.PACKET)
        W0, V0 = solver.initialize_longwave(U0)
        result = solver.run(U0, W0, V0, save_every=40, snapshot_times=[0.1])
        U = result["snapshots"][f"t_{0.1:.6f}"]["U"]
        ref = self._snapshot(2.0, 0.1)
        self.assertLess(float(np.abs(U - ref["U"]).max()), 1e-12)


if __name__ == "__main__":
    unittest.main()
