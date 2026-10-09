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

"""Experiment orchestration for the numerical results of Chapter 5.

Each experiment writes a self-contained folder ``outputs/expN_.../`` with a CSV
(and, where relevant, a ``summary.json`` and NPZ fields). ``run_all_experiments``
runs the whole suite; ``fgdss_plots.plot_all`` then turns the CSVs into figures.

Experiment map (numbering matches the output folders)
-----------------------------------------------------
1. ``run_operator_verification``      -- symmetry and negative-semidefiniteness
   of the Riesz matrices (Section 4.4).
2. ``run_source_free_conservation``   -- discrete mass / Hamiltonian conservation
   and momentum drift in time (Section 4.12).
3. ``run_time_convergence``           -- temporal self-convergence, observed order
   in tau (Sections 4.10-4.11).
4. ``run_spatial_convergence``        -- spatial self-convergence on nested grids,
   observed order in h.
5. ``run_fractional_order_sensitivity`` -- invariants across fractional orders.
6. ``run_integer_analytic_convergence`` -- integer-order (conventional) comparison
   against the exact linear Schrodinger standing wave.
7. ``run_momentum_residual_balance``  -- verifies dt J = R_disp + R_cub + R_lw and
   reports the non-vanishing residuals (Section 4.12).
8. ``run_boundary_domain_sweep``      -- boundary contamination vs domain size,
   probing the zero-extension Dirichlet treatment (Section 4.3).
9. ``run_performance_scaling``        -- cost of the reference direct solver vs the
   number of unknowns (Section 4.8).

Self-convergence (experiments 3, 4) is used for the *fractional* problem because
closed-form Riesz derivatives of localized profiles are not available; the exact
analytic benchmark of experiment 6 complements it at integer order.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List
import csv
import json
import platform
import time

import numpy as np
import scipy

from fgdss_fd_operators import FDGrid, Riesz2DCache, clear_riesz_1d_cache, vec
from fgdss_solver import (
    FGDSSParams,
    FGDSSSolver,
    gaussian_initial_data,
    add_relative_drifts,
    write_history_csv,
    save_run_npz,
)


def _write_rows(rows: List[Dict[str, float | int | str]], path: str | Path) -> None:
    """Write tabular experiment summaries with heterogeneous row dictionaries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    all_keys = set().union(*(row.keys() for row in rows))
    keys = list(rows[0].keys())
    keys += sorted(k for k in all_keys if k not in keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _summary_from_history(history: List[Dict[str, float]]) -> Dict[str, float]:
    """Summarize a diagnostic history.

    Returns the maximum and final relative drifts of ``M, E_theta, Jx, Jy``,
    the largest long-wave residuals, and the largest Picard iteration count.
    """
    hist = add_relative_drifts(history)
    out: Dict[str, float] = {}
    for key in ["M", "E_theta", "Jx", "Jy"]:
        out[f"max_RE_{key}"] = max(float(r[f"RE_{key}"]) for r in hist)
        out[f"final_RE_{key}"] = float(hist[-1][f"RE_{key}"])
    out["max_res_W"] = max(float(r.get("res_W", 0.0)) for r in hist)
    out["max_res_V"] = max(float(r.get("res_V", 0.0)) for r in hist)
    out["max_picard_iters"] = max(float(r.get("picard_iters", 0.0)) for r in hist)
    return out


# --------------------------------------------------------------------------- Exp 1

def run_operator_verification(output_root: str | Path) -> List[Dict[str, float]]:
    """Verify symmetry and negative-semidefinite behavior of Riesz matrices."""
    outdir = Path(output_root) / "exp1_operator_verification"
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []
    rng = np.random.default_rng(1234)
    for M in (16, 24, 32, 48):
        grid = FDGrid(M=M, N=M)
        ops = Riesz2DCache(grid)
        Z = rng.normal(size=(grid.nx, grid.ny))
        for order in (1.2, 1.5, 1.8, 2.0):
            for axis, A, A1 in (
                ("x", ops.Dx(order), ops.dx1d(order)),
                ("y", ops.Dy(order), ops.dy1d(order)),
            ):
                sym = A - A.T
                z = vec(Z)
                q = np.real(np.vdot(z, A @ z)) * grid.hx * grid.hy
                eigenvalues = np.linalg.eigvalsh(A1.toarray())
                rows.append(
                    {
                        "M": M,
                        "N": M,
                        "order": order,
                        "axis": axis,
                        "symmetry_fro_norm": float(np.sqrt((sym.multiply(sym)).sum())),
                        "quadratic_form": float(q),
                        "is_negative_semidefinite_sample": int(q <= 1e-10),
                        "min_eigenvalue_1d": float(eigenvalues[0]),
                        "max_eigenvalue_1d": float(eigenvalues[-1]),
                        "is_negative_semidefinite_spectrum": int(eigenvalues[-1] <= 1e-12),
                    }
                )
    _write_rows(rows, outdir / "operator_verification.csv")
    return rows


# --------------------------------------------------------------------------- Exp 2

def run_source_free_conservation(
    output_root: str | Path,
    grid: FDGrid,
    params: FGDSSParams,
    save_every: int = 10,
    snapshot_times: Iterable[float] = (0.0, 0.01, 0.025, 0.05),
    init_kwargs: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Experiment 2: evolve a Gaussian packet and monitor the invariants (Section 5.3).

    Writes ``diagnostics.csv``, the final fields, the snapshots, and
    ``summary.json`` to ``exp2_source_free_conservation/`` and returns the
    solver result.
    """
    outdir = Path(output_root) / "exp2_source_free_conservation"
    outdir.mkdir(parents=True, exist_ok=True)
    solver = FGDSSSolver(grid, params)
    U0 = gaussian_initial_data(grid, **(init_kwargs or {}))
    W0, V0 = solver.initialize_longwave(U0)
    result = solver.run(U0, W0, V0, save_every=save_every, snapshot_times=list(snapshot_times))
    history = add_relative_drifts(result["history"])  # type: ignore[arg-type]
    result["history"] = history
    write_history_csv(history, outdir / "diagnostics.csv")
    save_run_npz(result, outdir)
    summary = _summary_from_history(history)
    summary["wall_seconds"] = float(result["wall_seconds"])
    summary["final_abs_U_max"] = float(np.max(np.abs(result["final"]["U"])))  # type: ignore[index]
    summary["max_boundary_abs_u"] = max(float(r.get("boundary_max_abs_u", 0.0)) for r in history)
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return result


# --------------------------------------------------------------------------- Exp 3

def _l2_diff(A: np.ndarray, B: np.ndarray, grid: FDGrid) -> float:
    """Discrete L^2 norm of ``A - B`` on ``grid``."""
    return float(np.sqrt(grid.hx * grid.hy * np.sum(np.abs(A - B) ** 2)))


def run_time_convergence(
    output_root: str | Path,
    base_grid: FDGrid,
    base_params: FGDSSParams,
    dts: Iterable[float] = (1.6e-3, 8.0e-4, 4.0e-4, 2.0e-4, 1.0e-4),
    t_final: float = 1.6e-2,
) -> List[Dict[str, float]]:
    """Temporal self-convergence at fixed grid.

    All refinements share the same spatial discretization, so the spatial error
    cancels in the differences and the temporal order is isolated. The finest
    dt provides the reference solution at t_final; the error of each coarser dt
    is measured in the discrete L2 norm and the observed order is estimated by
    successive halving.
    """
    outdir = Path(output_root) / "exp3_time_convergence"
    outdir.mkdir(parents=True, exist_ok=True)
    dts = sorted((float(d) for d in dts), reverse=True)  # coarse -> fine

    finals: Dict[float, np.ndarray] = {}
    rows: List[Dict[str, float]] = []
    for dt in dts:
        params = replace(base_params, dt=dt, t_final=float(t_final))
        solver = FGDSSSolver(base_grid, params)
        U0 = gaussian_initial_data(base_grid)
        W0, V0 = solver.initialize_longwave(U0)
        result = solver.run(U0, W0, V0, save_every=max(1, int(round(t_final / dt))))
        finals[dt] = np.asarray(result["final"]["U"])  # type: ignore[index]
        hist = add_relative_drifts(result["history"])  # type: ignore[arg-type]
        summ = _summary_from_history(hist)
        rows.append(
            {
                "dt": dt,
                "Nt": int(round(t_final / dt)),
                "wall_seconds": float(result["wall_seconds"]),
                "max_RE_M": summ["max_RE_M"],
                "max_RE_E_theta": summ["max_RE_E_theta"],
            }
        )

    U_ref = finals[dts[-1]]  # finest
    for r in rows:
        dt = r["dt"]
        if dt == dts[-1]:
            r["err_L2"] = 0.0
        else:
            r["err_L2"] = _l2_diff(finals[dt], U_ref, base_grid)
    # observed order between consecutive (coarse->fine, excluding the reference)
    conv = [r for r in rows if r["dt"] != dts[-1]]
    for i in range(len(conv)):
        if i == 0:
            conv[i]["order"] = float("nan")
        else:
            e_prev, e_cur = conv[i - 1]["err_L2"], conv[i]["err_L2"]
            ratio = conv[i - 1]["dt"] / conv[i]["dt"]
            conv[i]["order"] = (
                float(np.log(e_prev / e_cur) / np.log(ratio)) if e_cur > 0 else float("nan")
            )
    _write_rows(rows, outdir / "time_convergence.csv")
    return rows


# --------------------------------------------------------------------------- Exp 4

def _restrict(U_fine: np.ndarray, ratio: int) -> np.ndarray:
    """Restrict a fine interior field to the coarse interior nodes.

    On a fixed box the coarse nodes are a subset of the fine nodes when the
    fine grid has ratio-times more subintervals, so the restriction is exact
    sampling (no interpolation error). Coarse interior index i maps to fine
    interior index (i+1)*ratio-1.
    """
    nxf, nyf = U_fine.shape
    nxc = (nxf + 1) // ratio - 1
    nyc = (nyf + 1) // ratio - 1
    ix = np.array([(i + 1) * ratio - 1 for i in range(nxc)])
    iy = np.array([(j + 1) * ratio - 1 for j in range(nyc)])
    return U_fine[np.ix_(ix, iy)]


def run_spatial_convergence(
    output_root: str | Path,
    base_params: FGDSSParams,
    Ms: Iterable[int] = (16, 32, 64),
    t_final: float = 2.0e-3,
    dt: float = 2.5e-4,
) -> List[Dict[str, float]]:
    """Spatial self-convergence on nested grids.

    ``Ms`` must be nested by factors of two so that coarse nodes are a subset of
    fine nodes and the restriction is exact. The time step is held small so that
    the temporal error is subdominant; the spatial order is estimated from the
    discrete L2 differences of consecutive nested solutions (three-grid
    Richardson).
    """
    outdir = Path(output_root) / "exp4_spatial_convergence"
    outdir.mkdir(parents=True, exist_ok=True)
    Ms = sorted(int(m) for m in Ms)  # coarse -> fine

    finals: Dict[int, np.ndarray] = {}
    grids: Dict[int, FDGrid] = {}
    rows: List[Dict[str, float]] = []
    for M in Ms:
        grid = FDGrid(M=M, N=M)
        params = replace(base_params, dt=float(dt), t_final=float(t_final))
        solver = FGDSSSolver(grid, params)
        U0 = gaussian_initial_data(grid)
        W0, V0 = solver.initialize_longwave(U0)
        result = solver.run(U0, W0, V0, save_every=max(1, int(round(t_final / dt))))
        finals[M] = np.asarray(result["final"]["U"])  # type: ignore[index]
        grids[M] = grid
        hist = add_relative_drifts(result["history"])  # type: ignore[arg-type]
        summ = _summary_from_history(hist)
        rows.append(
            {
                "M": M,
                "hx": grid.hx,
                "wall_seconds": float(result["wall_seconds"]),
                "max_RE_M": summ["max_RE_M"],
                "max_RE_E_theta": summ["max_RE_E_theta"],
            }
        )

    # consecutive nested-pair errors: ||U_M - restrict(U_{2M})||_2 on the M grid
    for i, M in enumerate(Ms):
        r = rows[i]
        if i + 1 < len(Ms) and Ms[i + 1] % M == 0:
            ratio = Ms[i + 1] // M
            U_fine_restricted = _restrict(finals[Ms[i + 1]], ratio)
            r["err_L2"] = _l2_diff(finals[M], U_fine_restricted, grids[M])
        else:
            r["err_L2"] = float("nan")
    # observed order from consecutive pair errors
    for i in range(len(Ms)):
        if i == 0 or not np.isfinite(rows[i].get("err_L2", np.nan)) or not np.isfinite(
            rows[i - 1].get("err_L2", np.nan)
        ):
            rows[i]["order"] = float("nan")
        else:
            ratio = Ms[i] / Ms[i - 1]
            e_prev, e_cur = rows[i - 1]["err_L2"], rows[i]["err_L2"]
            rows[i]["order"] = (
                float(np.log(e_prev / e_cur) / np.log(ratio)) if e_cur > 0 else float("nan")
            )
    _write_rows(rows, outdir / "spatial_convergence.csv")
    return rows


# --------------------------------------------------------------------------- Exp 5

def run_fractional_order_sensitivity(
    output_root: str | Path,
    grid: FDGrid,
    base_params: FGDSSParams,
    orders: Iterable[float] = (2.0, 1.8, 1.5, 1.2),
    t_final: float = 2.0e-2,
) -> List[Dict[str, float]]:
    """Experiment 5: invariant defects for a common fractional order (Section 5.7).

    For each ``s`` in ``orders`` all six orders are set to ``s`` and a short
    run records the maximum drifts and the wall time.
    """
    outdir = Path(output_root) / "exp5_fractional_order_sensitivity"
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []
    for s in orders:
        params = replace(base_params, lam=s, mu=s, nu=s, kappa=s, rho=s, sigma=s, t_final=t_final)
        solver = FGDSSSolver(grid, params)
        U0 = gaussian_initial_data(grid)
        W0, V0 = solver.initialize_longwave(U0)
        result = solver.run(U0, W0, V0, save_every=max(1, int(round(t_final / params.dt / 10))))
        hist = add_relative_drifts(result["history"])  # type: ignore[arg-type]
        finalU = result["final"]["U"]  # type: ignore[index]
        summ = _summary_from_history(hist)
        rows.append(
            {
                "order": float(s),
                "max_abs_U_final": float(np.max(np.abs(finalU))),
                "l2_abs_U_final": float(np.sqrt(grid.hx * grid.hy * np.sum(np.abs(finalU) ** 2))),
                "wall_seconds": float(result["wall_seconds"]),
                **summ,
            }
        )
    _write_rows(rows, outdir / "fractional_order_sensitivity.csv")
    return rows


# --------------------------------------------------------------------------- Exp 6

def linear_schrodinger_eigenmode(grid: FDGrid, t: float, alpha: float, beta: float, p: int = 1, q: int = 1) -> np.ndarray:
    """Closed-form solution of the integer-order (conventional) linear problem.

    At lambda=2 the Ortigueira operator reduces to the classical 3-point
    Laplacian, so with gamma=xi=0 the FGDSS short-wave equation becomes the
    linear Schrodinger equation i u_t + alpha u_xx + beta u_yy = 0. On the box
    with homogeneous Dirichlet data the standing wave
        u(x,y,t) = e^{-i omega t} sin(p pi (x-xL)/Lx) sin(q pi (y-yL)/Ly),
        omega = alpha (p pi/Lx)^2 + beta (q pi/Ly)^2,
    is an exact analytical solution used here as a convergence benchmark.
    """
    X, Y = grid.X, grid.Y
    Lx, Ly = grid.xR - grid.xL, grid.yR - grid.yL
    kx, ky = p * np.pi / Lx, q * np.pi / Ly
    omega = alpha * kx ** 2 + beta * ky ** 2
    return np.exp(-1j * omega * t) * np.sin(kx * (X - grid.xL)) * np.sin(ky * (Y - grid.yL))


def run_integer_analytic_convergence(
    output_root: str | Path,
    base_params: FGDSSParams,
    Ms: Iterable[int] = (16, 32, 64),
    t_final: float = 0.2,
    dt: float = 1.0e-3,
    domain: float = 10.0,
) -> List[Dict[str, float]]:
    """Integer-order (conventional-case) convergence against an exact analytic
    solution.

    The fractional orders are set to the integer value 2, the coupling and
    nonlinearity are switched off (gamma=xi=0), and the numerical solution is
    compared with the closed-form linear Schrodinger standing wave
    (linear_schrodinger_eigenmode). The spatial order of accuracy is estimated
    from the discrete L2 error at t_final; the time step is held small so the
    spatial error dominates.
    """
    outdir = Path(output_root) / "exp6_integer_analytic_convergence"
    outdir.mkdir(parents=True, exist_ok=True)
    Ms = sorted(int(m) for m in Ms)
    rows: List[Dict[str, float]] = []
    for M in Ms:
        grid = FDGrid(xL=-domain, xR=domain, yL=-domain, yR=domain, M=M, N=M)
        params = replace(
            base_params,
            gamma=0.0, xi=0.0,
            lam=2.0, mu=2.0, nu=2.0, kappa=2.0, rho=2.0, sigma=2.0,
            dt=float(dt), t_final=float(t_final),
        )
        solver = FGDSSSolver(grid, params)
        U0 = linear_schrodinger_eigenmode(grid, 0.0, params.alpha, params.beta).astype(complex)
        W0, V0 = solver.initialize_longwave(U0)
        result = solver.run(U0, W0, V0, save_every=10 ** 9)
        U_exact = linear_schrodinger_eigenmode(grid, t_final, params.alpha, params.beta)
        err = _l2_diff(np.asarray(result["final"]["U"]), U_exact, grid)  # type: ignore[index]
        rows.append(
            {
                "M": M,
                "hx": grid.hx,
                "err_L2": err,
                "wall_seconds": float(result["wall_seconds"]),
            }
        )
    for i in range(len(rows)):
        if i == 0:
            rows[i]["order"] = float("nan")
        else:
            ratio = rows[i]["M"] / rows[i - 1]["M"]
            e_prev, e_cur = rows[i - 1]["err_L2"], rows[i]["err_L2"]
            rows[i]["order"] = (
                float(np.log(e_prev / e_cur) / np.log(ratio)) if e_cur > 0 else float("nan")
            )
    _write_rows(rows, outdir / "integer_analytic_convergence.csv")
    return rows


# --------------------------------------------------------------------------- Exp 7

def run_momentum_residual_balance(
    output_root: str | Path,
    grid: FDGrid,
    base_params: FGDSSParams,
    t_final: float = 1.0e-2,
) -> List[Dict[str, float]]:
    """Verify the discrete momentum-balance decomposition of Section 4.12.

    At each step the three residuals R_disp, R_cub, R_lw are evaluated and the
    identity dt J_z = R_disp + R_cub + R_lw is checked. The residuals do not
    vanish (no exact discrete momentum conservation), while the balance defect
    is at the level of the nonlinear-solver tolerance.
    """
    outdir = Path(output_root) / "exp7_momentum_residual_balance"
    outdir.mkdir(parents=True, exist_ok=True)
    params = replace(base_params, t_final=float(t_final))
    solver = FGDSSSolver(grid, params)
    U0 = gaussian_initial_data(grid)
    W0, V0 = solver.initialize_longwave(U0)
    U, W, V = U0.copy(), W0.copy(), V0.copy()
    Nt = int(round(t_final / params.dt))

    rows: List[Dict[str, float]] = []
    for n in range(Nt):
        Un, Wn, Vn, _ = solver.picard_step(U, W, V, return_info=True)
        terms = solver.momentum_residual_terms(U, Un, W, Wn, V, Vn)
        rows.append({"n": n + 1, "t": (n + 1) * params.dt, **terms})
        U, W, V = Un, Wn, Vn

    _write_rows(rows, outdir / "momentum_residual_balance.csv")
    summary = {
        "max_balance_defect": max(
            max(abs(r["balance_defect_x"]), abs(r["balance_defect_y"])) for r in rows
        ),
        "max_abs_R_disp": max(max(abs(r["R_disp_x"]), abs(r["R_disp_y"])) for r in rows),
        "max_abs_R_cub": max(max(abs(r["R_cub_x"]), abs(r["R_cub_y"])) for r in rows),
        "max_abs_R_lw": max(max(abs(r["R_lw_x"]), abs(r["R_lw_y"])) for r in rows),
        "max_abs_dtJ": max(max(abs(r["dtJ_x"]), abs(r["dtJ_y"])) for r in rows),
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return rows


# --------------------------------------------------------------------------- Exp 8

def run_boundary_domain_sweep(
    output_root: str | Path,
    base_params: FGDSSParams,
    half_widths: Iterable[float] = (8.0, 12.0, 16.0, 24.0),
    h: float = 1.0,
    window_half_width: float = 8.0,
    t_final: float = 2.0e-2,
) -> List[Dict[str, float]]:
    """Controlled domain-size sweep for the zero-extension boundary treatment.

    The mesh width is fixed, so ``M=2L/h`` grows with the box.  Every box samples
    the same physical initial field: an unwindowed Gaussian multiplied by a
    compact, domain-independent cosine-squared window supported in
    ``[-window_half_width, window_half_width]^2``.  The final solution on each
    smaller box is also compared on matching nodes with the largest-box result.
    This separates the domain effect from changes in resolution and initial data.
    """
    outdir = Path(output_root) / "exp8_boundary_domain_sweep"
    outdir.mkdir(parents=True, exist_ok=True)
    half_widths = sorted(float(L) for L in half_widths)
    if h <= 0.0:
        raise ValueError("h must be positive.")
    if not (0.0 < window_half_width <= half_widths[0]):
        raise ValueError("window_half_width must be positive and no larger than the smallest box.")

    rows: List[Dict[str, float]] = []
    grids: Dict[float, FDGrid] = {}
    finals: Dict[float, np.ndarray] = {}
    for L in half_widths:
        M_float = 2.0 * L / float(h)
        M = int(round(M_float))
        if not np.isclose(M, M_float):
            raise ValueError(f"The box L={L:g} is not commensurate with h={h:g}.")
        grid = FDGrid(xL=-L, xR=L, yL=-L, yR=L, M=M, N=M)
        params = replace(base_params, t_final=float(t_final))
        solver = FGDSSSolver(grid, params)
        U0 = gaussian_initial_data(grid, use_window=False)
        R = float(window_half_width)
        Wx = np.where(np.abs(grid.X) < R, np.cos(0.5 * np.pi * grid.X / R) ** 2, 0.0)
        Wy = np.where(np.abs(grid.Y) < R, np.cos(0.5 * np.pi * grid.Y / R) ** 2, 0.0)
        U0 = U0 * Wx * Wy
        W0, V0 = solver.initialize_longwave(U0)
        result = solver.run(U0, W0, V0, save_every=1)
        hist = add_relative_drifts(result["history"])  # type: ignore[arg-type]
        max_bd = max(float(r.get("boundary_max_abs_u", 0.0)) for r in hist)
        summ = _summary_from_history(hist)
        grids[L] = grid
        finals[L] = np.asarray(result["final"]["U"])  # type: ignore[index]
        rows.append(
            {
                "half_width": L,
                "M": M,
                "hx": grid.hx,
                "dt": params.dt,
                "window_half_width": R,
                "max_boundary_abs_u": max_bd,
                "max_RE_M": summ["max_RE_M"],
                "max_RE_E_theta": summ["max_RE_E_theta"],
                "max_RE_Jx": summ["max_RE_Jx"],
                "wall_seconds": float(result["wall_seconds"]),
            }
        )

    L_ref = half_widths[-1]
    grid_ref = grids[L_ref]
    U_ref = finals[L_ref]
    x_lookup = {round(float(x), 12): i for i, x in enumerate(grid_ref.x)}
    y_lookup = {round(float(y), 12): j for j, y in enumerate(grid_ref.y)}
    for row in rows:
        L = float(row["half_width"])
        grid = grids[L]
        ix = np.array([x_lookup[round(float(x), 12)] for x in grid.x], dtype=int)
        iy = np.array([y_lookup[round(float(y), 12)] for y in grid.y], dtype=int)
        U_ref_on_box = U_ref[np.ix_(ix, iy)]
        row["err_to_largest_L2"] = _l2_diff(finals[L], U_ref_on_box, grid)

    _write_rows(rows, outdir / "boundary_domain_sweep.csv")
    return rows


# --------------------------------------------------------------------------- Exp 9

def run_performance_scaling(
    output_root: str | Path,
    base_params: FGDSSParams,
    Ms: Iterable[int] = (16, 24, 32, 48, 64),
    t_final: float = 5.0e-3,
) -> List[Dict[str, float | str]]:
    """Computational-cost scaling of the reference direct-solver implementation.

    Reports the complete solver setup time, the per-step wall time and the
    average number of Picard iterations as functions of the number of interior
    unknowns. This documents the dense/direct cost of the reference solver and
    motivates the Toeplitz/FFT/Krylov acceleration discussed in Section 4.8.
    """
    outdir = Path(output_root) / "exp9_performance_scaling"
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float | str]] = []
    for M in Ms:
        # Measure complete cold setup rather than reusing operators assembled by
        # earlier experiments in the same Python process.
        clear_riesz_1d_cache()
        grid = FDGrid(M=int(M), N=int(M))
        params = replace(base_params, t_final=float(t_final))
        t0 = time.perf_counter()
        solver = FGDSSSolver(grid, params)
        setup_seconds = time.perf_counter() - t0
        U0 = gaussian_initial_data(grid)
        W0, V0 = solver.initialize_longwave(U0)
        result = solver.run(U0, W0, V0, save_every=10 ** 9)  # no intermediate diagnostics
        Nt = int(round(t_final / params.dt))
        picard_hist = result["picard_history"]  # type: ignore[assignment]
        picard_iters = [float(r["picard_iters"]) for r in picard_hist]
        rows.append(
            {
                "M": int(M),
                "Ntot": grid.ntot,
                "setup_seconds": float(setup_seconds),
                "wall_seconds": float(result["wall_seconds"]),
                "per_step_ms": float(result["wall_seconds"]) / max(1, Nt) * 1.0e3,
                "mean_picard_iters": float(np.mean(picard_iters)) if picard_iters else float("nan"),
                "operator_cache": "cold",
            }
        )
    _write_rows(rows, outdir / "performance_scaling.csv")
    return rows


# --------------------------------------------------------------------------- all

def run_all_experiments(
    output_root: str | Path,
    grid: FDGrid | None = None,
    params: FGDSSParams | None = None,
) -> Dict[str, object]:
    """Run the reference nine-experiment suite reported in Chapter 5."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    grid = FDGrid(M=48, N=48) if grid is None else grid
    params = FGDSSParams(dt=5.0e-4, t_final=2.0e-2) if params is None else params

    meta: Dict[str, object] = {
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "grid": asdict(grid),
        "params": asdict(params),
    }
    meta["operator_verification"] = run_operator_verification(output_root)
    meta["source_free"] = "see exp2_source_free_conservation"
    # exp2 is the primary conservation run: a longer, genuinely dynamic evolution
    # (stronger packet, T=1) so that mass/energy conservation is demonstrated over
    # a real evolution rather than a near-static snapshot. The same run supplies
    # the 3-D field/amplitude figures.
    cons_params = replace(params, dt=2.5e-3, t_final=1.0)
    run_source_free_conservation(
        output_root,
        grid=grid,
        params=cons_params,
        save_every=max(1, int(round(cons_params.t_final / cons_params.dt / 10))),
        snapshot_times=(0.0, cons_params.t_final / 2.0, cons_params.t_final),
        init_kwargs={"amplitude": 0.6, "sigma_x": 3.5, "sigma_y": 3.5, "kx0": 0.6, "ky0": 0.3},
    )
    # exp3/exp4: genuine convergence-order studies (self-convergence).
    time_params = replace(params, picard_tol=1.0e-13)
    meta["time_convergence"] = run_time_convergence(
        output_root, FDGrid(M=24, N=24), time_params
    )
    meta["spatial_convergence"] = run_spatial_convergence(output_root, params)
    meta["fractional_sensitivity"] = run_fractional_order_sensitivity(output_root, grid, params, t_final=1.0e-2)
    # exp6: integer-order (conventional) comparison against an exact analytic solution.
    meta["integer_analytic_convergence"] = run_integer_analytic_convergence(
        output_root, params, Ms=(16, 32, 64), t_final=0.05, dt=2.5e-3
    )
    # exp7: momentum-balance residual decomposition (Section 4.12).
    meta["momentum_residual_balance"] = run_momentum_residual_balance(
        output_root, FDGrid(M=32, N=32), params, t_final=1.0e-2
    )
    # exp8: boundary / domain-size sweep (zero-extension Dirichlet treatment).
    meta["boundary_domain_sweep"] = run_boundary_domain_sweep(
        output_root, params, half_widths=(8.0, 12.0, 16.0, 24.0), h=1.0,
        window_half_width=8.0, t_final=1.0e-2
    )
    # exp9: computational-cost scaling of the reference direct solver.
    meta["performance_scaling"] = run_performance_scaling(output_root, params, Ms=(16, 24, 32, 48))
    with open(output_root / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
