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

"""Small regression tests for operators, invariants, and momentum balance."""

from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import gamma

from fgdss_fd_operators import FDGrid, Riesz2DCache, ortigueira_coefficients, riesz_1d
from fgdss_solver import FGDSSParams, FGDSSSolver, gaussian_initial_data, save_run_npz


class OrtigueiraCoefficientTests(unittest.TestCase):
    """Ortigueira coefficients: closed form, large offsets, integer order."""
    def test_recurrence_matches_closed_form_at_moderate_offsets(self) -> None:
        """The stable recurrence agrees with the Gamma-function closed form."""
        order = 1.3
        got = ortigueira_coefficients(order, 40)
        expected = np.array(
            [
                (-1.0) ** j
                * gamma(order + 1.0)
                / (gamma(order / 2.0 - j + 1.0) * gamma(order / 2.0 + j + 1.0))
                for j in range(41)
            ]
        )
        np.testing.assert_allclose(got, expected, rtol=2.0e-13, atol=1.0e-15)

    def test_large_offsets_do_not_become_false_zeros(self) -> None:
        """Coefficients stay finite and nonzero beyond the Gamma overflow range."""
        coeffs = ortigueira_coefficients(1.2, 220)
        self.assertTrue(np.all(np.isfinite(coeffs)))
        self.assertNotEqual(float(coeffs[170]), 0.0)
        self.assertNotEqual(float(coeffs[220]), 0.0)

    def test_integer_order_recovers_three_point_stencil(self) -> None:
        """At order 2 the coefficients reduce to the stencil (2, -1, 0, ...)."""
        coeffs = ortigueira_coefficients(2.0, 8)
        np.testing.assert_array_equal(coeffs[:2], np.array([2.0, -1.0]))
        np.testing.assert_array_equal(coeffs[2:], np.zeros(7))


class DiscreteStructureTests(unittest.TestCase):
    """Structure of the discrete operators and of one solver step."""
    def test_riesz_matrix_is_symmetric_negative_semidefinite(self) -> None:
        """The 1-D Riesz matrix is exactly symmetric and negative semidefinite."""
        D = riesz_1d(15, 0.5, 1.2).toarray()
        np.testing.assert_allclose(D, D.T, rtol=0.0, atol=0.0)
        self.assertLessEqual(float(np.linalg.eigvalsh(D).max()), 1.0e-12)

    def test_integer_full_order_does_not_make_all_coupled_blocks_sparse(self) -> None:
        """At order 2 the dispersive operator is sparse, but the mixed operator is not."""
        grid = FDGrid(xL=-4.0, xR=4.0, yL=-4.0, yR=4.0, M=8, N=8)
        ops = Riesz2DCache(grid)
        self.assertLess(ops.Dx(2.0).nnz, ops.Dx(1.0).nnz)
        self.assertGreater(ops.Dxy(2.0).nnz, ops.Dx(2.0).nnz)

    def test_one_step_preserves_mass_energy_and_closes_momentum_balance(self) -> None:
        """One converged step conserves mass and energy and closes the momentum balance."""
        grid = FDGrid(xL=-8.0, xR=8.0, yL=-8.0, yR=8.0, M=10, N=10)
        params = FGDSSParams(
            dt=1.0e-4,
            t_final=1.0e-4,
            picard_tol=1.0e-12,
            picard_max_iter=50,
        )
        solver = FGDSSSolver(grid, params)
        U0 = gaussian_initial_data(grid, amplitude=0.15)
        W0, V0 = solver.initialize_longwave(U0)
        U1, W1, V1, info = solver.picard_step(U0, W0, V0, return_info=True)

        self.assertEqual(info["picard_converged"], 1.0)
        mass_scale = 1.0 + abs(solver.mass(U0))
        energy_scale = 1.0 + abs(solver.energy(U0, W0, V0))
        self.assertLess(abs(solver.mass(U1) - solver.mass(U0)) / mass_scale, 1.0e-12)
        self.assertLess(
            abs(solver.energy(U1, W1, V1) - solver.energy(U0, W0, V0)) / energy_scale,
            1.0e-12,
        )

        residuals = solver.momentum_residual_terms(U0, U1, W0, W1, V0, V1)
        self.assertLess(abs(residuals["balance_defect_x"]), 1.0e-10)
        self.assertLess(abs(residuals["balance_defect_y"]), 1.0e-10)

    def test_under_relaxation_is_rejected(self) -> None:
        """A Picard relaxation other than 1 is rejected."""
        with self.assertRaises(ValueError):
            FGDSSParams(picard_relaxation=0.8)

    def test_nonconverged_picard_step_is_rejected(self) -> None:
        """A step whose Picard iteration does not converge raises an error."""
        self.assertTrue(FGDSSParams().fail_on_nonconvergence)
        grid = FDGrid(xL=-4.0, xR=4.0, yL=-4.0, yR=4.0, M=8, N=8)
        params = FGDSSParams(
            dt=1.0e-2,
            picard_tol=1.0e-30,
            picard_max_iter=1,
            fail_on_nonconvergence=True,
        )
        solver = FGDSSSolver(grid, params)
        U0 = gaussian_initial_data(grid, amplitude=0.2)
        W0, V0 = solver.initialize_longwave(U0)
        with self.assertRaises(RuntimeError):
            solver.picard_step(U0, W0, V0)

    def test_invalid_model_and_run_parameters_are_rejected(self) -> None:
        """Invalid orders, steps, tolerances, and output intervals are rejected."""
        invalid_params = (
            {"lam": 1.0},
            {"dt": 0.0},
            {"t_final": -1.0},
            {"picard_tol": 0.0},
            {"picard_max_iter": 0},
        )
        for kwargs in invalid_params:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FGDSSParams(**kwargs)

        grid = FDGrid(xL=-1.0, xR=1.0, yL=-1.0, yR=1.0, M=4, N=4)
        solver = FGDSSSolver(grid, FGDSSParams(t_final=0.0))
        with self.assertRaises(ValueError):
            solver.run(np.zeros((grid.nx, grid.ny), dtype=complex), save_every=0)

    def test_snapshot_export_removes_obsolete_generated_files(self) -> None:
        """Saving a run removes stale snapshot files and lists the current ones."""
        grid = FDGrid(xL=-1.0, xR=1.0, yL=-1.0, yR=1.0, M=4, N=4)
        params = FGDSSParams()
        zeros = np.zeros((grid.nx, grid.ny))
        fields = {"U": zeros.astype(complex), "W": zeros, "V": zeros, "Q": zeros}
        result = {
            "grid": grid,
            "params": params,
            "final": fields,
            "snapshots": {"t_0.000000": fields, "t_1.000000": fields},
            "wall_seconds": 0.0,
        }
        with TemporaryDirectory() as tmp:
            snapdir = Path(tmp) / "snapshots"
            snapdir.mkdir()
            np.savez_compressed(snapdir / "t_0.250000.npz", U=zeros)
            save_run_npz(result, tmp)
            self.assertFalse((snapdir / "t_0.250000.npz").exists())
            manifest = json.loads((snapdir / "snapshot_manifest.json").read_text())
            self.assertEqual(
                manifest["files"], ["t_0.000000.npz", "t_1.000000.npz"]
            )


if __name__ == "__main__":
    unittest.main()
