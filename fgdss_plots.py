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

"""Plot helpers for FDM + Picard FGDSS experiment outputs.

The style is tuned for a printed thesis: a Computer-Modern-like math font so the
figures match the LaTeX body, a colorblind-safe categorical palette
(Okabe--Ito), recessive grid/axes, and vector (PDF) plus raster (PNG) output.
Convergence figures annotate the observed order; sweeps that mix quantities of
different scale use small multiples rather than a second y-axis.
"""
from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe--Ito colorblind-safe categorical palette, assigned in fixed order.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#000000", "#F0E442"]
_GUIDE = "#9AA0A6"  # recessive grey for reference-slope guides


def _apply_style() -> None:
    """Apply the thesis plotting style: serif text, Computer Modern math,
    Okabe--Ito colors, light grid, no top or right spines.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.titlesize": 11,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


_apply_style()

# Prettified axis labels for raw CSV keys.
_XLABELS = {
    "dt": r"$\tau$",
    "hx": r"$h_x$",
    "M": r"$M$",
    "order": "fractional order $s$",
    "half_width": r"domain half-width $L$",
    "Ntot": r"$N_{\mathrm{tot}}$",
    "t": r"$t$",
}


def _save(fig, outdir: Path, stem: str) -> None:
    """Save ``fig`` as ``<stem>.pdf`` and ``<stem>.png`` in ``outdir`` and close it."""
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def _read_csv(path: str | Path):
    """Read a CSV file into a list of dictionaries."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fit_order(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares log--log slope (observed order of accuracy)."""
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])


# --------------------------------------------------------------------------- exp2

def plot_diagnostic_drifts(csv_path: str | Path, output_dir: str | Path) -> None:
    """Relative drifts of ``M_h, E_{theta,h}, J_{x,h}, J_{y,h}`` versus time
    (experiment 2, Section 5.3).
    """
    rows = _read_csv(csv_path)
    if not rows:
        return
    t = np.array([float(r["t"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    series = (("RE_M", r"$M_h$"), ("RE_E_theta", r"$E_{\theta,h}$"), ("RE_Jx", r"$J_{x,h}$"), ("RE_Jy", r"$J_{y,h}$"))
    for key, label in series:
        if key in rows[0]:
            y = np.array([float(r[key]) for r in rows])
            ax.semilogy(t, np.maximum(y, 1e-18), marker="o", label=label)
    ax.axhline(np.finfo(float).eps, color=_GUIDE, ls=":", lw=1.0, label=r"machine $\epsilon$")
    ax.set_xlabel(_XLABELS["t"])
    ax.set_ylabel("relative drift $\\mathrm{RE}$")
    ax.legend(ncol=2)
    _save(fig, Path(output_dir), "diagnostic_relative_drifts")


def plot_field_surface_3d(
    npz_path: str | Path,
    output_dir: str | Path,
    field: str = "abs_U",
    stem: str | None = None,
    zlabel: str = r"$|U|$",
    title: str | None = None,
) -> None:
    """3-D surface plot of a single field (default: the short-wave amplitude).

    The height and the (sequential, colorblind-safe) viridis color both encode
    magnitude, which is the appropriate encoding for an amplitude.
    """
    data = np.load(npz_path)
    x, y = data["x"], data["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = np.real(data[field])
    fig = plt.figure(figsize=(7.0, 5.2))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(
        X, Y, Z, cmap="viridis", rcount=120, ccount=120, linewidth=0, antialiased=True
    )
    ax.set_xlabel(r"$x$", labelpad=6)
    ax.set_ylabel(r"$y$", labelpad=6)
    ax.set_zlabel(zlabel, labelpad=4)
    ax.view_init(elev=28, azim=-125)
    ax.grid(False)
    if title:
        ax.set_title(title)
    cb = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.10)
    cb.ax.tick_params(labelsize=8)
    _save(fig, Path(output_dir), stem or f"{field}_surface3d")


def plot_fields_surface_grid(npz_path: str | Path, output_dir: str | Path, stem: str = "fields_surface3d") -> None:
    """2x2 grid of 3-D surfaces of the short-wave amplitude and the long-wave
    fields ``|U|, W, V, Q``."""
    data = np.load(npz_path)
    x, y = data["x"], data["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    fields = [("abs_U", r"$|U|$"), ("W", r"$W$"), ("V", r"$V$"), ("Q", r"$Q$")]
    fig = plt.figure(figsize=(9.6, 7.8))
    for i, (key, title) in enumerate(fields):
        if key not in data:
            continue
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        Z = np.real(data[key])
        surf = ax.plot_surface(
            X, Y, Z, cmap="viridis", rcount=80, ccount=80, linewidth=0, antialiased=True
        )
        ax.set_title(title)
        ax.set_xlabel(r"$x$", labelpad=2)
        ax.set_ylabel(r"$y$", labelpad=2)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=28, azim=-125)
        ax.grid(False)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.10)
    fig.tight_layout()
    _save(fig, Path(output_dir), stem)


def plot_final_fields(npz_path: str | Path, output_dir: str | Path) -> None:
    """Contour maps of ``|U|, W, V, Q`` at the final time (Section 5.3).

    The signed fields use a diverging color scale centered at zero.
    """
    data = np.load(npz_path)
    x, y = data["x"], data["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    fields = [("abs_U", r"$|U|$"), ("W", r"$W$"), ("V", r"$V$"), ("Q", r"$Q$")]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.9), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, fields):
        if key not in data:
            ax.set_visible(False)
            continue
        f = np.real(data[key])
        if key == "abs_U":
            cf = ax.contourf(X, Y, f, levels=40, cmap="viridis")
        else:
            # v4: diverging map centred at zero for the signed long-wave fields.
            m = float(np.abs(f).max()) or 1.0
            cf = ax.contourf(X, Y, f, levels=40, cmap="RdBu_r",
                             norm=matplotlib.colors.TwoSlopeNorm(0.0, -m, m))
        cb = fig.colorbar(cf, ax=ax, shrink=0.9)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_aspect("equal")
        ax.grid(False)
    _save(fig, Path(output_dir), "final_fields_contours")


def _peak_indices(a: np.ndarray):
    """Index pair of the maximum entry of a 2-D array."""
    return np.unravel_index(np.argmax(a), a.shape)


def plot_pulse_profiles(snapshot_dir: str | Path, output_dir: str | Path,
                        times=("0.000000", "0.500000", "1.000000")) -> None:
    """v4: cuts of |U| through the packet peak at several times (linear and log)."""
    snapshot_dir = Path(snapshot_dir)
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), constrained_layout=True)
    for k, (t, c) in enumerate(zip(times, OKABE_ITO)):
        z = np.load(snapshot_dir / f"t_{t}.npz")
        x, y, a = z["x"], z["y"], z["abs_U"]
        i, j = _peak_indices(a)
        lab = rf"$t={float(t):g}$"
        axes[0].plot(x, a[:, j], color=c, label=lab, marker="o", ms=2.5)
        axes[1].plot(y, a[i, :], color=c, label=lab, marker="o", ms=2.5)
        if k in (0, len(times) - 1):
            axes[2].semilogy(y, np.maximum(a[i, :], 1e-12), color=c, label=lab, marker="o", ms=2.5)
    axes[0].set_xlabel(r"$x$"); axes[0].set_ylabel(r"$|U|$ along the peak row")
    axes[1].set_xlabel(r"$y$"); axes[1].set_ylabel(r"$|U|$ along the peak column")
    axes[2].set_xlabel(r"$y$"); axes[2].set_ylabel(r"$|U|$ (log scale)")
    axes[2].set_ylim(1e-6, 1)
    for ax, lab in zip(axes, "abc"):
        ax.set_xlim(-20, 20)
        ax.set_title(f"({lab})")
        ax.legend()
    _save(fig, Path(output_dir), "pulse_profiles")


def plot_longwave_profiles(npz_path: str | Path, output_dir: str | Path) -> None:
    """v4: cuts of W, V, Q and |U|^2 through the packet peak."""
    z = np.load(npz_path)
    x, y = z["x"], z["y"]
    i, j = _peak_indices(z["abs_U"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True, sharey=True)
    for key, lab, c in (("W", r"$W$", OKABE_ITO[0]), ("V", r"$V$", OKABE_ITO[1]), ("Q", r"$Q$", OKABE_ITO[2])):
        axes[0].plot(x, z[key][:, j], color=c, label=lab)
        axes[1].plot(y, z[key][i, :], color=c, label=lab)
    axes[0].plot(x, z["abs_U"][:, j] ** 2, color="k", ls="--", lw=1.2, label=r"$|U|^2$")
    axes[1].plot(y, z["abs_U"][i, :] ** 2, color="k", ls="--", lw=1.2, label=r"$|U|^2$")
    axes[0].set_xlabel(r"$x$ (cut through the peak)")
    axes[1].set_xlabel(r"$y$ (cut through the peak)")
    for ax in axes:
        ax.axhline(0, color=_GUIDE, lw=0.8)
        ax.set_xlim(-20, 20)
    axes[0].legend()
    _save(fig, Path(output_dir), "longwave_profiles")


# --------------------------------------------------------------------------- exp3/4/6

def plot_convergence(csv_path: str | Path, xkey: str, output_dir: str | Path, stem: str, expected_slope: int = 2) -> None:
    """Log--log convergence plot of the discrete L2 error, annotating the
    observed order and drawing an offset reference-slope guide."""
    rows = _read_csv(csv_path)
    rows = [r for r in rows if r.get("err_L2", "") not in ("", "nan") and float(r["err_L2"]) > 0]
    if len(rows) < 2:
        return
    h = np.array([float(r["dt" if xkey == "dt" else "hx"]) for r in rows])
    e = np.array([float(r["err_L2"]) for r in rows])
    order = _fit_order(h, e)

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.loglog(h, e, marker="o", color=OKABE_ITO[0], label=r"$\|U_h-U_{\mathrm{ref}}\|_2$")
    # reference slope, offset above the data so it does not overlap the markers
    ref = 2.2 * e.max() * (h / h.max()) ** expected_slope
    ax.loglog(h, ref, ls="--", lw=1.4, color=_GUIDE, label=fr"reference slope ${expected_slope}$")
    ax.set_xlabel(_XLABELS.get(xkey, xkey))
    ax.set_ylabel(r"discrete $L^2$ error")
    if np.isfinite(order):
        ax.annotate(
            fr"observed order $\approx {order:.2f}$",
            xy=(0.05, 0.08), xycoords="axes fraction", ha="left", va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=_GUIDE, lw=0.8, alpha=0.9),
        )
    ax.legend(loc="lower right")
    _save(fig, Path(output_dir), stem)


# --------------------------------------------------------------------------- exp5

def plot_summary_lines(csv_path: str | Path, xkey: str, output_dir: str | Path, stem: str) -> None:
    """Maximum relative drift of each diagnostic against a swept parameter
    ``xkey`` (for example the common order of experiment 5).
    """
    rows = _read_csv(csv_path)
    if not rows:
        return
    x = np.array([float(r[xkey]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    series = (("max_RE_M", r"$M_h$"), ("max_RE_E_theta", r"$E_{\theta,h}$"), ("max_RE_Jx", r"$J_{x,h}$"), ("max_RE_Jy", r"$J_{y,h}$"))
    for key, label in series:
        if key in rows[0]:
            vals = np.array([float(r[key]) for r in rows])
            ax.semilogy(x, np.maximum(vals, 1e-18), marker="o", label=label)
    ax.set_xlabel(_XLABELS.get(xkey, xkey))
    ax.set_ylabel("maximum relative drift")
    ax.legend(ncol=2)
    _save(fig, Path(output_dir), stem)


# --------------------------------------------------------------------------- exp7

def plot_momentum_residuals(csv_path: str | Path, output_dir: str | Path) -> None:
    """Magnitudes of ``delta_t J_x``, the residuals ``R_disp, R_cub, R_lw``, and
    the balance defect versus time (experiment 7, Section 5.8).
    """
    rows = _read_csv(csv_path)
    if not rows:
        return
    t = np.array([float(r["t"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    series = (
        ("dtJ_x", r"$\delta_t J_{x,h}$"),
        ("R_disp_x", r"$\mathcal{R}_{\mathrm{disp}}$"),
        ("R_cub_x", r"$\mathcal{R}_{\mathrm{cub}}$"),
        ("R_lw_x", r"$\mathcal{R}_{\mathrm{lw}}$"),
        ("balance_defect_x", r"balance defect"),
    )
    for key, label in series:
        if key in rows[0]:
            y = np.array([abs(float(r[key])) for r in rows])
            ax.semilogy(t, np.maximum(y, 1e-18), marker="o", label=label)
    ax.set_xlabel(_XLABELS["t"])
    ax.set_ylabel(r"magnitude ($x$-component)")
    ax.legend(ncol=2)
    _save(fig, Path(output_dir), "momentum_residual_balance")


# --------------------------------------------------------------------------- exp8

def plot_boundary_sweep(csv_path: str | Path, output_dir: str | Path) -> None:
    """Controlled fixed-mesh domain sweep: boundary/solution measures and drifts."""
    rows = _read_csv(csv_path)
    if not rows:
        return
    L = np.array([float(r["half_width"]) for r in rows])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)

    bd = np.array([float(r["max_boundary_abs_u"]) for r in rows])
    err = np.array([float(r["err_to_largest_L2"]) for r in rows])
    ax0.semilogy(L, np.maximum(bd, 1e-18), marker="o", color=OKABE_ITO[1], label=r"outer-ring $|U|$")
    ax0.semilogy(L, np.maximum(err, 1e-18), marker="s", color=OKABE_ITO[0], label=r"$\|U_L-U_{24}\|_2$")
    ax0.set_xlabel(_XLABELS["half_width"])
    ax0.set_ylabel("boundary / solution measure")
    ax0.set_title("(a) controlled domain comparison")
    ax0.legend()

    diagnostic_series = (("max_RE_M", r"$M_h$"), ("max_RE_E_theta", r"$E_{\theta,h}$"), ("max_RE_Jx", r"$J_{x,h}$"))
    for key, label in diagnostic_series:
        if key in rows[0]:
            y = np.array([float(r[key]) for r in rows])
            ax1.semilogy(L, np.maximum(y, 1e-18), marker="o", label=label)
    ax1.axhline(np.finfo(float).eps, color=_GUIDE, ls=":", lw=1.0, label=r"machine $\epsilon$")
    ax1.set_xlabel(_XLABELS["half_width"])
    ax1.set_ylabel("maximum relative drift")
    ax1.set_title("(b) invariants and momentum diagnostic")
    ax1.legend(ncol=2)
    _save(fig, Path(output_dir), "boundary_domain_sweep")


# --------------------------------------------------------------------------- exp9

def plot_performance_scaling(csv_path: str | Path, output_dir: str | Path) -> None:
    """Log--log cost scaling vs number of unknowns, with fitted exponents."""
    rows = _read_csv(csv_path)
    if not rows:
        return
    ntot = np.array([float(r["Ntot"]) for r in rows])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)

    for ax, key, title, color in (
        (ax0, "per_step_ms", "(a) per-step wall time", OKABE_ITO[0]),
        (ax1, "setup_seconds", "(b) complete solver setup", OKABE_ITO[2]),
    ):
        y = np.array([float(r[key]) for r in rows])
        p = _fit_order(ntot, y)
        ax.loglog(ntot, y, marker="o", color=color)
        if np.isfinite(p):
            ref = y.min() * (ntot / ntot.min()) ** p
            ax.loglog(ntot, ref, ls="--", lw=1.2, color=_GUIDE,
                      label=fr"$\propto N_{{\mathrm{{tot}}}}^{{{p:.2f}}}$")
            ax.legend(loc="upper left")
        ax.set_xlabel(_XLABELS["Ntot"])
        ax.set_ylabel("ms" if key == "per_step_ms" else "s")
        ax.set_title(title)
    _save(fig, Path(output_dir), "performance_scaling")


# --------------------------------------------------------------------------- driver

def plot_all(output_root: str | Path) -> None:
    """Draw every figure of experiments 1-9 whose input exists under
    ``output_root``; figures go to ``output_root/figures/expN/``.
    """
    root = Path(output_root)
    figroot = root / "figures"
    diag = root / "exp2_source_free_conservation" / "diagnostics.csv"
    if diag.exists():
        plot_diagnostic_drifts(diag, figroot / "exp2")
    final = root / "exp2_source_free_conservation" / "final_fields.npz"
    if final.exists():
        plot_final_fields(final, figroot / "exp2")
        plot_field_surface_3d(final, figroot / "exp2", field="abs_U", stem="absU_surface3d")
        plot_fields_surface_grid(final, figroot / "exp2", stem="fields_surface3d")
        plot_longwave_profiles(final, figroot / "exp2")
    snaps = root / "exp2_source_free_conservation" / "snapshots"
    if (snaps / "t_1.000000.npz").exists():
        plot_pulse_profiles(snaps, figroot / "exp2")
    tconv = root / "exp3_time_convergence" / "time_convergence.csv"
    if tconv.exists():
        plot_convergence(tconv, "dt", figroot / "exp3", "time_convergence", expected_slope=2)
    sconv = root / "exp4_spatial_convergence" / "spatial_convergence.csv"
    if sconv.exists():
        plot_convergence(sconv, "hx", figroot / "exp4", "spatial_convergence", expected_slope=2)
    fsens = root / "exp5_fractional_order_sensitivity" / "fractional_order_sensitivity.csv"
    if fsens.exists():
        plot_summary_lines(fsens, "order", figroot / "exp5", "fractional_order_drifts")
    aconv = root / "exp6_integer_analytic_convergence" / "integer_analytic_convergence.csv"
    if aconv.exists():
        plot_convergence(aconv, "hx", figroot / "exp6", "integer_analytic_convergence", expected_slope=2)
    mom = root / "exp7_momentum_residual_balance" / "momentum_residual_balance.csv"
    if mom.exists():
        plot_momentum_residuals(mom, figroot / "exp7")
    bnd = root / "exp8_boundary_domain_sweep" / "boundary_domain_sweep.csv"
    if bnd.exists():
        plot_boundary_sweep(bnd, figroot / "exp8")
    perf = root / "exp9_performance_scaling" / "performance_scaling.csv"
    if perf.exists():
        plot_performance_scaling(perf, figroot / "exp9")
