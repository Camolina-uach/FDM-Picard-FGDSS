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

"""Experiment 12 (thesis v4, Section 5.7): shape of the pulse for different
common fractional orders.

The conservation run of Section 5.3 (grid M = N = 48 on [-20, 20]^2,
tau = 2.5e-3, T = 1, packet A = 0.6, sigma_x = sigma_y = 3.5,
(k_x, k_y) = (0.6, 0.3), coefficients of Table 5.1) is repeated with all six
orders set to a common value s in {2.0, 1.8, 1.5, 1.2}. The run with s = 1.2 is
the conservation run itself and is compared with the archived exp2 fields.
Snapshots every 0.1 time units give the peak amplitude and the centroid of
|U|^2 as functions of time.

Usage::

    python exp12_fractional_order_profiles.py run S [OUTPUT_ROOT]   # one order
    python exp12_fractional_order_profiles.py summarize [OUTPUT_ROOT]
    python exp12_fractional_order_profiles.py surfaces [OUTPUT_ROOT]   # field figures only

``run`` writes ``OUTPUT_ROOT/exp12_fractional_order_profiles/s_S/`` (diagnostics,
final fields, snapshots); ``summarize`` writes ``pulse_statistics.csv`` and the
figures ``OUTPUT_ROOT/figures/exp12/fractional_order_profiles``,
``fractional_order_surfaces3d`` (log10|U|), ``fractional_order_fields_maps``,
and ``fractional_order_fields_surfaces3d`` (|U|, W, V), each as PDF and PNG.
The default OUTPUT_ROOT is ``outputs``.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from fgdss_fd_operators import FDGrid
from fgdss_solver import (FGDSSParams, FGDSSSolver, add_relative_drifts,
                          gaussian_initial_data, save_run_npz, write_history_csv)

HERE = Path(__file__).resolve().parent
ORDERS = (2.0, 1.8, 1.5, 1.2)
PACKET = {"amplitude": 0.6, "sigma_x": 3.5, "sigma_y": 3.5, "kx0": 0.6, "ky0": 0.3}
SNAP_TIMES = tuple(round(0.1 * k, 10) for k in range(11))


def _root(arg: str | None) -> Path:
    """Resolve the output root: ``outputs/`` by default, relative paths from this folder."""
    root = Path(arg) if arg else HERE / "outputs"
    return root if root.is_absolute() else HERE / root


def run_order(s: float, output_root: Path) -> Path:
    """Repeat the conservation run with all six orders equal to ``s``.

    Writes the diagnostics, the final fields, and the snapshots to
    ``exp12_fractional_order_profiles/s_<s>/`` and returns that folder.
    """
    grid = FDGrid(M=48, N=48)
    params = FGDSSParams(dt=2.5e-3, t_final=1.0, lam=s, mu=s, nu=s, kappa=s, rho=s, sigma=s)
    solver = FGDSSSolver(grid, params)
    U0 = gaussian_initial_data(grid, **PACKET)
    W0, V0 = solver.initialize_longwave(U0)
    result = solver.run(U0, W0, V0, save_every=40, snapshot_times=list(SNAP_TIMES))
    outdir = output_root / "exp12_fractional_order_profiles" / f"s_{s:.1f}"
    history = add_relative_drifts(result["history"])
    write_history_csv(history, outdir / "diagnostics.csv")
    save_run_npz(result, outdir)
    print(f"s={s:.1f}: wall {result['wall_seconds']:.1f} s, "
          f"max RE_M {max(r['RE_M'] for r in history):.2e}, "
          f"max RE_E {max(r['RE_E_theta'] for r in history):.2e}")
    return outdir


def pulse_stats(z) -> dict:
    """Peak of |U|, centroid and RMS radius of |U|^2 on the grid."""
    x, y, a = z["x"], z["y"], z["abs_U"]
    rho = a**2
    m = rho.sum()
    X, Y = np.meshgrid(x, y, indexing="ij")
    xc, yc = (X * rho).sum() / m, (Y * rho).sum() / m
    r2 = (((X - xc) ** 2 + (Y - yc) ** 2) * rho).sum() / m
    return {"peak": float(a.max()), "xc": float(xc), "yc": float(yc), "rms_radius": float(np.sqrt(r2))}


def plot_surfaces(output_root: Path, floor: float = 1e-5) -> None:
    """2x2 surfaces of log10|U| at t = 1, one per order, on common axes.

    On a linear scale the four surfaces are indistinguishable (the peaks differ
    by less than 1%); the logarithmic scale shows the dispersive tails. Values
    below ``floor`` are clipped.
    """
    import fgdss_plots as fp   # applies the thesis plotting style
    plt = fp.plt
    base = output_root / "exp12_fractional_order_profiles"
    zmin = np.log10(floor)
    fig = plt.figure(figsize=(9.6, 7.8))
    surf = None
    for i, s in enumerate(ORDERS):
        z = np.load(base / f"s_{s:.1f}" / "snapshots" / f"t_{1.0:.6f}.npz")
        X, Y = np.meshgrid(z["x"], z["y"], indexing="ij")
        L = np.log10(np.maximum(z["abs_U"], floor))
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        surf = ax.plot_surface(X, Y, L, cmap="viridis", vmin=zmin, vmax=0.0,
                               rcount=96, ccount=96, linewidth=0, antialiased=True)
        ax.set_zlim(zmin, 0.0)
        ax.set_title(fr"$s={s:g}$")
        ax.set_xlabel(r"$x$", labelpad=2)
        ax.set_ylabel(r"$y$", labelpad=2)
        ax.set_zlabel(r"$\log_{10}|U|$", labelpad=2)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=28, azim=-125)
        ax.grid(False)
    fig.tight_layout()
    fig.colorbar(surf, ax=fig.axes, shrink=0.6, pad=0.04, label=r"$\log_{10}|U|$")
    fp._save(fig, output_root / "figures" / "exp12", "fractional_order_surfaces3d")


def plot_fields_by_order(output_root: Path) -> None:
    """|U|, W, V at t = 1 for every order: 4x3 color maps and 4x3 surfaces.

    Rows are the orders and columns the fields; each column shares one color
    scale (and one height range in 3-D), so the panels can be compared down a
    column. The signed fields W and V use a diverging scale centered at zero.
    """
    import matplotlib.colors as mcolors
    import fgdss_plots as fp   # applies the thesis plotting style
    plt = fp.plt
    base = output_root / "exp12_fractional_order_profiles"
    data = {s: np.load(base / f"s_{s:.1f}" / "snapshots" / f"t_{1.0:.6f}.npz") for s in ORDERS}
    fields = (("abs_U", r"$|U|$"), ("W", r"$W$"), ("V", r"$V$"))
    vmax = {k: max(float(np.abs(np.real(data[s][k])).max()) for s in ORDERS) for k, _ in fields}
    vmin = {k: min(float(np.real(data[s][k]).min()) for s in ORDERS) for k, _ in fields}
    outdir = output_root / "figures" / "exp12"

    # Color maps.
    fig, axes = plt.subplots(len(ORDERS), 3, figsize=(9.0, 11.6), constrained_layout=True)
    for j, (key, label) in enumerate(fields):
        if key == "abs_U":
            cmap, norm = "viridis", mcolors.Normalize(0.0, vmax[key])
        else:
            cmap, norm = "RdBu_r", mcolors.TwoSlopeNorm(0.0, -vmax[key], vmax[key])
        for i, s in enumerate(ORDERS):
            z = data[s]
            X, Y = np.meshgrid(z["x"], z["y"], indexing="ij")
            ax = axes[i, j]
            cf = ax.contourf(X, Y, np.real(z[key]), levels=40, cmap=cmap, norm=norm)
            ax.set_aspect("equal")
            ax.grid(False)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.set_ylabel(fr"$s={s:g}$" + "\n" + r"$y$")
            if i == len(ORDERS) - 1:
                ax.set_xlabel(r"$x$")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=axes[:, j], orientation="horizontal", shrink=0.85, pad=0.01, aspect=30)
    fp._save(fig, outdir, "fractional_order_fields_maps")

    # 3-D surfaces.
    fig = plt.figure(figsize=(9.6, 12.4))
    for i, s in enumerate(ORDERS):
        z = data[s]
        X, Y = np.meshgrid(z["x"], z["y"], indexing="ij")
        for j, (key, label) in enumerate(fields):
            ax = fig.add_subplot(len(ORDERS), 3, 3 * i + j + 1, projection="3d")
            lo = 0.0 if key == "abs_U" else vmin[key]
            ax.plot_surface(X, Y, np.real(z[key]), cmap="viridis", vmin=lo, vmax=vmax[key],
                            rcount=72, ccount=72, linewidth=0, antialiased=True)
            ax.set_zlim(lo, vmax[key])
            ax.set_title(fr"{label}, $s={s:g}$", fontsize=10)
            ax.set_xlabel(r"$x$", labelpad=-2, fontsize=8)
            ax.set_ylabel(r"$y$", labelpad=-2, fontsize=8)
            ax.tick_params(labelsize=6, pad=0)
            ax.view_init(elev=28, azim=-125)
            ax.grid(False)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=0.97, wspace=0.0, hspace=0.12)
    fp._save(fig, outdir, "fractional_order_fields_surfaces3d")

    # Numbers quoted in the text.
    for s in ORDERS:
        z = data[s]
        print(f"s={s:g}: max|U| {z['abs_U'].max():.3f}, W [{np.real(z['W']).min():.4f}, "
              f"{np.real(z['W']).max():.4f}], V [{np.real(z['V']).min():.4f}, {np.real(z['V']).max():.4f}]")


def summarize(output_root: Path) -> None:
    """Collect the four runs: pulse statistics, all figures, and the check
    of the ``s = 1.2`` run against the archived conservation run.
    """
    base = output_root / "exp12_fractional_order_profiles"
    rows = []
    for s in ORDERS:
        for t in SNAP_TIMES:
            z = np.load(base / f"s_{s:.1f}" / "snapshots" / f"t_{t:.6f}.npz")
            rows.append({"s": s, "t": t, **pulse_stats(z)})
    with (base / "pulse_statistics.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    import fgdss_plots as fp   # applies the thesis plotting style
    plt = fp.plt
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    z0 = np.load(base / f"s_{ORDERS[0]:.1f}" / "snapshots" / f"t_{0.0:.6f}.npz")
    a0 = z0["abs_U"]
    i0, j0 = np.unravel_index(a0.argmax(), a0.shape)
    axes[0, 0].plot(z0["x"], a0[:, j0], color="k", ls="--", lw=1.2, label=r"$t=0$")
    axes[0, 1].semilogy(z0["y"], np.maximum(a0[i0, :], 1e-12), color="k", ls="--", lw=1.2, label=r"$t=0$")
    for s, c in zip(ORDERS, fp.OKABE_ITO):
        z = np.load(base / f"s_{s:.1f}" / "snapshots" / f"t_{1.0:.6f}.npz")
        a = z["abs_U"]
        i, j = np.unravel_index(a.argmax(), a.shape)
        axes[0, 0].plot(z["x"], a[:, j], color=c, label=fr"$s={s:g}$")
        axes[0, 1].semilogy(z["y"], np.maximum(a[i, :], 1e-12), color=c, label=fr"$s={s:g}$")
        sel = [r for r in rows if r["s"] == s]
        tt = [r["t"] for r in sel]
        axes[1, 0].plot(tt, [r["peak"] for r in sel], color=c, marker="o", ms=3, label=fr"$s={s:g}$")
        axes[1, 1].plot([r["xc"] for r in sel], [r["yc"] for r in sel], color=c, marker="o", ms=3,
                        label=fr"$s={s:g}$")
    axes[0, 0].set_xlabel(r"$x$"); axes[0, 0].set_ylabel(r"$|U|$ at $t=1$ (peak row)")
    axes[0, 0].set_xlim(-12, 12)
    axes[0, 1].set_xlabel(r"$y$"); axes[0, 1].set_ylabel(r"$|U|$ at $t=1$ (peak column, log scale)")
    axes[0, 1].set_xlim(-20, 20); axes[0, 1].set_ylim(1e-6, 1)
    axes[1, 0].set_xlabel(r"$t$"); axes[1, 0].set_ylabel(r"$\max|U|$")
    axes[1, 1].set_xlabel(r"centroid $x_c$"); axes[1, 1].set_ylabel(r"centroid $y_c$")
    for ax, lab in zip(axes.flat, "abcd"):
        ax.set_title(f"({lab})")
        ax.legend()
    fp._save(fig, output_root / "figures" / "exp12", "fractional_order_profiles")

    plot_surfaces(output_root)
    plot_fields_by_order(output_root)

    # Check: the s = 1.2 run reproduces the archived conservation run (exp2).
    arch = HERE / "outputs_revision" / "exp2_source_free_conservation" / "final_fields.npz"
    if arch.exists():
        za = np.load(arch)
        zn = np.load(base / "s_1.2" / "final_fields.npz")
        diff = float(np.abs(za["U"] - zn["U"]).max())
        with (base / "check_against_exp2.json").open("w", encoding="utf-8") as fh:
            json.dump({"max_abs_diff_U_final_vs_exp2": diff}, fh, indent=2)
        print(f"s=1.2 vs archived exp2: max |U - U_exp2| = {diff:.2e}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        run_order(float(sys.argv[2]), _root(sys.argv[3] if len(sys.argv) > 3 else None))
    elif len(sys.argv) >= 2 and sys.argv[1] == "summarize":
        summarize(_root(sys.argv[2] if len(sys.argv) > 2 else None))
    elif len(sys.argv) >= 2 and sys.argv[1] == "surfaces":
        root = _root(sys.argv[2] if len(sys.argv) > 2 else None)
        plot_surfaces(root)
        plot_fields_by_order(root)
    else:
        sys.exit(__doc__)
