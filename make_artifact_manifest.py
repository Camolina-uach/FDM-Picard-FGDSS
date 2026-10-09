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

"""Create the configuration and SHA-256 manifest for the reviewed artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import scipy


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = HERE / "outputs_revision"


def _sha256(path: Path) -> str:
    """SHA-256 digest of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configuration() -> dict[str, object]:
    """Reviewed configuration of every experiment, as recorded in the manifest."""
    coefficients = {
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "xi": 1.0,
        "psi": 1.0,
        "eta": 2.0,
        "phi": 2.0,
        "chi": 1.0,
        "theta": 1.0,
    }
    default_orders = {
        "lambda": 1.2,
        "mu": 1.2,
        "nu": 1.2,
        "kappa": 1.2,
        "rho": 1.2,
        "sigma": 1.2,
    }
    default_packet = {
        "amplitude": 0.25,
        "sigma_x": 4.5,
        "sigma_y": 3.0,
        "center": [0.0, 0.0],
        "wave_numbers": [0.35, 0.20],
        "window": "domain sin-squared window",
    }
    picard = {
        "tolerance": 1.0e-10,
        "maximum_iterations": 30,
        "relaxation": 1.0,
    }
    return {
        "common": {
            "coefficients": coefficients,
            "default_orders": default_orders,
            "default_packet": default_packet,
            "picard": picard,
            "boundary_realization": "homogeneous Dirichlet by zero extension",
        },
        "exp1_operator_verification": {
            "status": "regenerated in final environment",
            "grids_M_equals_N": [16, 24, 32, 48],
            "orders": [1.2, 1.5, 1.8, 2.0],
            "directions": ["x", "y"],
            "checks": ["symmetry", "full 1D spectrum", "sample quadratic form"],
        },
        "exp2_source_free_conservation": {
            "status": "retained and checked long run",
            "domain": [-20.0, 20.0, -20.0, 20.0],
            "M_equals_N": 48,
            "dt": 0.0025,
            "t_final": 1.0,
            "coefficients": coefficients,
            "orders": default_orders,
            "packet": {
                "amplitude": 0.6,
                "sigma_x": 3.5,
                "sigma_y": 3.5,
                "center": [0.0, 0.0],
                "wave_numbers": [0.6, 0.3],
                "window": "domain sin-squared window",
            },
            "diagnostic_stride_steps": 40,
            "active_snapshot_times": [0.0, 0.5, 1.0],
            "retained_run_fail_on_nonconvergence": False,
            "saved_diagnostics_all_converged": True,
        },
        "exp3_time_convergence": {
            "status": "regenerated in final environment",
            "domain": [-20.0, 20.0, -20.0, 20.0],
            "M_equals_N": 24,
            "dt": [0.0016, 0.0008, 0.0004, 0.0002, 0.0001],
            "t_final": 0.016,
            "picard_tolerance": 1.0e-13,
            "packet": default_packet,
        },
        "exp4_spatial_convergence": {
            "status": "retained and checked run",
            "domain": [-20.0, 20.0, -20.0, 20.0],
            "grids_M_equals_N": [16, 32, 64],
            "dt": 0.00025,
            "t_final": 0.002,
            "restriction": "exact sampling on nested nodes",
            "packet": default_packet,
        },
        "exp5_fractional_order_sensitivity": {
            "status": "retained and checked run",
            "domain": [-20.0, 20.0, -20.0, 20.0],
            "M_equals_N": 48,
            "dt": 0.0005,
            "t_final": 0.01,
            "common_orders": [2.0, 1.8, 1.5, 1.2],
            "packet": default_packet,
            "diagnostic_stride_steps": 2,
        },
        "exp6_integer_analytic_convergence": {
            "status": "retained and checked run",
            "domain": [-10.0, 10.0, -10.0, 10.0],
            "grids_M_equals_N": [16, 32, 64],
            "dt": 0.0025,
            "t_final": 0.05,
            "orders": "all set to 2",
            "gamma": 0.0,
            "xi": 0.0,
            "exact_mode_indices": [1, 1],
        },
        "exp7_momentum_residual_balance": {
            "status": "retained and checked run",
            "domain": [-20.0, 20.0, -20.0, 20.0],
            "M_equals_N": 32,
            "dt": 0.0005,
            "t_final": 0.01,
            "packet": default_packet,
            "diagnostics": "every time step",
        },
        "exp8_boundary_domain_sweep": {
            "status": "regenerated in final environment",
            "half_widths": [8.0, 12.0, 16.0, 24.0],
            "M_equals_N": [16, 24, 32, 48],
            "h": 1.0,
            "dt": 0.0005,
            "t_final": 0.01,
            "packet": {
                **default_packet,
                "window": "common compact cosine-squared window",
                "window_half_width": 8.0,
            },
            "diagnostics": "every time step",
        },
        "exp9_performance_scaling": {
            "status": "regenerated in final environment",
            "domain": [-20.0, 20.0, -20.0, 20.0],
            "grids_M_equals_N": [16, 24, 32, 48],
            "dt": 0.0005,
            "t_final": 0.005,
            "runs_per_grid": 1,
            "operator_cache": "cleared before each timed construction",
            "thread_setting": "numerical-library defaults",
            "packet": default_packet,
        },
        "exp10_manufactured_longwave": {
            "status": "thesis v4 addition (Section 5.6)",
            "domain": [-12.0, 12.0, -12.0, 12.0],
            "grids_M_equals_N": [16, 24, 32, 48, 64],
            "common_orders": [1.2, 1.5, 1.8, 2.0],
            "long_wave_coefficients": {"psi": 1.0, "eta": 2.0, "phi": 2.0, "chi": 1.0, "theta": 1.0},
            "fields": "w* = p(x/a) e(y/a), v* = e(x/a) p(y/a), e(u) = exp(-u^2), p(u) = u exp(-u^2), a = 2",
            "sources": "closed-form Riesz derivatives (confluent hypergeometric 1F1, mpmath)",
            "sensitivity": "solve with theta = 0 in M_h, sources computed with theta = 1",
            "environment": "Python 3.12.10, numpy 2.5.2, scipy 1.18.1, mpmath 1.3.0, Windows 11",
        },
        "exp11_longwave_matrix_spectrum": {
            "status": "thesis v4 addition (Section 5.1)",
            "quantity": "smallest eigenvalue of -M_h (eigsh, shift-invert at 0)",
            "cases": "all grids, domains, and common orders used in Chapter 5",
            "environment": "Python 3.12.10, numpy 2.5.2, scipy 1.18.1, Windows 11",
        },
        "exp12_fractional_order_profiles": {
            "status": "thesis v4 addition (Section 5.7)",
            "domain": [-20.0, 20.0, -20.0, 20.0],
            "M": 48,
            "N": 48,
            "dt": 0.0025,
            "t_final": 1.0,
            "common_orders": [2.0, 1.8, 1.5, 1.2],
            "packet": {"amplitude": 0.6, "sigma_x": 3.5, "sigma_y": 3.5, "kx0": 0.6, "ky0": 0.3},
            "snapshot_times": "every 0.1 time units",
            "figures": "profiles/peak/centroid (2x2); log10|U| surfaces at t = 1, clipped at 1e-5; maps and surfaces of |U|, W, V per order (4x3)",
            "check": "s = 1.2 run reproduces the archived exp2 final field (8.5e-14)",
            "environment": "Python 3.12.10, numpy 2.5.2, scipy 1.18.1, matplotlib 3.11.1, Windows 11",
        },
    }


def build_manifest(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    """Hash the active code and artifacts and write ``artifact_manifest.json``.

    Files under ``archive_excluded_snapshots`` are hashed separately and do not
    support the thesis. Returns the path of the manifest.
    """
    output_root = output_root.resolve()
    active_artifacts: list[Path] = []
    for path in output_root.rglob("*"):
        if (
            not path.is_file()
            or path.name in {"artifact_manifest.json", ".DS_Store"}
        ):
            continue
        if "archive_excluded_snapshots" in path.parts:
            continue
        active_artifacts.append(path)

    core_names = (
        "fgdss_fd_operators.py",
        "fgdss_babaoglu.py",
        "fgdss_solver.py",
        "fgdss_experiments.py",
        "fgdss_plots.py",
        "fgdss_tables.py",
        "run_fgdss.py",
        "export_thesis_assets.py",
        "make_artifact_manifest.py",
        "README.md",
        "requirements.txt",
        "requirements-lock.txt",
        "CITATION.cff",
        "CHANGELOG.md",
        ".zenodo.json",
        ".gitignore",
        ".gitattributes",
        "LICENSE",
        "tests/__init__.py",
        "tests/test_fgdss.py",
        "exp10_manufactured_longwave.py",
        "exp11_longwave_matrix_spectrum.py",
        "exp12_fractional_order_profiles.py",
        "tests/test_longwave_v4.py",
    )
    active_artifacts.extend(HERE / name for name in core_names)

    hashes = {
        path.relative_to(HERE).as_posix(): _sha256(path)
        for path in sorted(set(active_artifacts))
    }
    excluded_hashes = {
        path.relative_to(HERE).as_posix(): _sha256(path)
        for path in sorted((output_root / "archive_excluded_snapshots").rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "configuration": _configuration(),
        "active_artifact_sha256": hashes,
        "excluded_archive_sha256": excluded_hashes,
        "policy": (
            "Only active artifacts support the revised dissertation. Files in "
            "archive_excluded_snapshots are retained solely for provenance."
        ),
        "provenance_note": (
            "Manifest of release 1.1.0 (final version of the thesis, 2026-10-09). The 'environment' "
            "field describes the machine that generated this manifest. The exp1-exp9 "
            "artifacts are unchanged from the v3 snapshot (see requirements-lock.txt); "
            "exp10, exp11, and exp12 were added for v4 in the environment recorded "
            "in their configuration entries; the three v4 figures of exp2 were drawn "
            "from the archived exp2 snapshots."
        ),
    }
    target = output_root / "artifact_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_manifest())
