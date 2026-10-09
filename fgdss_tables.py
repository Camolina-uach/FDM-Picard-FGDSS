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

"""Generate the LaTeX data tables of the experiments chapter from the outputs.

Each function reads an experiment's CSV / ``summary.json`` and writes a
self-contained ``tabular`` environment (column spec + header + rows) to
``outputs/tables/<name>.tex``. The thesis chapter keeps the surrounding
``table`` float, caption and label and pulls in the numbers with

    \\input{tables_chapter4/<name>}

so that re-running ``run_fgdss.py`` refreshes every number without touching the
prose. Only the *data* tables are generated here; the reference-configuration
table is authored by hand in the chapter, since it records deliberate choices.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import List, Dict


# --------------------------------------------------------------------------- fmt

def _sci(x: float, sig: int = 1) -> str:
    """Format a real number as ``\\(m.mm\\times10^{e}\\)`` (or ``\\(0\\)``)."""
    x = float(x)
    if not math.isfinite(x):
        return "---"
    if x == 0.0:
        return r"\(0\)"
    neg = x < 0
    x = abs(x)
    e = math.floor(math.log10(x))
    m = x / 10.0 ** e
    m = round(m, sig)
    if m >= 10.0:
        m /= 10.0
        e += 1
    sign = "-" if neg else ""
    return rf"\({sign}{m:.{sig}f}\times10^{{{e}}}\)"


def _order(x) -> str:
    """Observed order with two decimals in math mode, or ``---`` if not finite."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "---"
    return "---" if not math.isfinite(v) else f"\\({v:.2f}\\)"


def _fixed(x, dec: int) -> str:
    """Fixed-point number with ``dec`` decimals in math mode."""
    return f"\\({float(x):.{dec}f}\\)"


def _int(x) -> str:
    """Integer in math mode."""
    return f"\\({int(round(float(x)))}\\)"


def _read(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dictionaries."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _wrap(colspec: str, header: str, rows: List[str]) -> str:
    """Wrap table rows in a ``booktabs`` tabular with the given column spec and header."""
    body = "\n".join(rows)
    return (
        f"\\begin{{tabular}}{{{colspec}}}\n"
        f"\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n"
        f"\\end{{tabular}}\n"
    )


def _write(text: str, path: Path) -> None:
    """Write ``text`` to ``path``, creating the parent folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------- individual tables

def conservation_table(root: Path, out: Path, n_times: int = 5) -> None:
    """Invariant defects at ``n_times`` equally spaced times (experiment 2, Section 5.3)."""
    rows = _read(root / "exp2_source_free_conservation" / "diagnostics.csv")
    nz = [r for r in rows if float(r["t"]) > 0]
    T = max(float(r["t"]) for r in nz)
    picked = []
    for i in range(1, n_times + 1):
        target = T * i / n_times
        picked.append(min(nz, key=lambda r: abs(float(r["t"]) - target)))
    body = [
        f"{_fixed(r['t'],1)} & {_sci(r['RE_M'])} & {_sci(r['RE_E_theta'])} & "
        f"{_sci(r['RE_Jx'])} & {_sci(r['RE_Jy'])} \\\\"
        for r in picked
    ]
    header = r"\(t\) & \(\mathrm{RE}_{M}\) & \(\mathrm{RE}_{E_\theta}\) & \(\mathrm{RE}_{J_x}\) & \(\mathrm{RE}_{J_y}\)"
    _write(_wrap("lllll", header, body), out / "exp2_conservation_table.tex")


def time_convergence_table(root: Path, out: Path) -> None:
    """Temporal self-convergence errors and observed orders (experiment 3, Section 5.4)."""
    rows = _read(root / "exp3_time_convergence" / "time_convergence.csv")
    rows.sort(key=lambda r: float(r["dt"]), reverse=True)
    body = []
    for r in rows:
        err = r["err_L2"]
        errcell = "(reference)" if (err in ("", "nan") or float(err) == 0.0) else _sci(err, 2)
        body.append(f"{_sci(r['dt'])} & {_int(r['Nt'])} & {errcell} & {_order(r.get('order'))} \\\\")
    header = r"\(\tau\) & \(N_t\) & \(\|U_\tau-U_{\mathrm{ref}}\|_2\) & order"
    _write(_wrap("llll", header, body), out / "exp3_time_convergence_table.tex")


def spatial_convergence_table(root: Path, out: Path) -> None:
    """Spatial self-convergence errors and observed orders (experiment 4, Section 5.4)."""
    rows = _read(root / "exp4_spatial_convergence" / "spatial_convergence.csv")
    rows.sort(key=lambda r: int(r["M"]))
    body = []
    for r in rows:
        err = r["err_L2"]
        errcell = "(reference)" if (err in ("", "nan")) else _sci(err, 2)
        body.append(f"{_int(r['M'])} & {_fixed(r['hx'],3)} & {errcell} & {_order(r.get('order'))} \\\\")
    header = r"\(M=N\) & \(h_x\) & \(\|U_h-U_{h/2}\|_2\) & order"
    _write(_wrap("llll", header, body), out / "exp4_spatial_convergence_table.tex")


def analytic_convergence_table(root: Path, out: Path) -> None:
    """Error against the exact standing wave at integer order (experiment 6, Section 5.5)."""
    rows = _read(root / "exp6_integer_analytic_convergence" / "integer_analytic_convergence.csv")
    rows.sort(key=lambda r: int(r["M"]))
    body = [
        f"{_int(r['M'])} & {_fixed(r['hx'],4)} & {_sci(r['err_L2'],2)} & {_order(r.get('order'))} \\\\"
        for r in rows
    ]
    header = r"\(M=N\) & \(h_x\) & \(\|U_h-u_{\mathrm{exact}}\|_2\) & order"
    _write(_wrap("llll", header, body), out / "exp6_analytic_convergence_table.tex")


def fractional_table(root: Path, out: Path) -> None:
    """Invariant defects, momentum drift, and wall time per common order (experiment 5, Section 5.7)."""
    rows = _read(root / "exp5_fractional_order_sensitivity" / "fractional_order_sensitivity.csv")
    rows.sort(key=lambda r: float(r["order"]), reverse=True)
    body = []
    for r in rows:
        wall_seconds = r.get("wall_seconds", r.get("cpu_seconds"))
        body.append(
            f"{_fixed(r['order'],1)} & {_sci(r['max_RE_M'])} & {_sci(r['max_RE_E_theta'])} & "
            f"{_sci(r['max_RE_Jx'])} & {_fixed(wall_seconds,1)} \\\\"
        )
    header = (r"\(s\) & \(\max\mathrm{RE}_{M}\) & \(\max\mathrm{RE}_{E_\theta}\) & "
              r"\(\max\mathrm{RE}_{J_x}\) & wall time (s)")
    _write(_wrap("lllll", header, body), out / "exp5_fractional_table.tex")


def momentum_table(root: Path, out: Path) -> None:
    """Terms of the discrete momentum balance (experiment 7, Section 5.8)."""
    rows = _read(root / "exp7_momentum_residual_balance" / "momentum_residual_balance.csv")
    picked = [rows[0], rows[len(rows) // 2], rows[-1]]
    body = [
        f"{_fixed(r['t'],4)} & {_sci(r['dtJ_x'],6)} & {_sci(r['R_disp_x'],6)} & "
        f"{_sci(r['R_cub_x'],6)} & {_sci(r['R_lw_x'],6)} & {_sci(r['balance_defect_x'],3)} \\\\"
        for r in picked
    ]
    header = (r"\(t\) & \(\delta_t J_{x,h}\) & \(\mathcal{R}_{\mathrm{disp}}\) & "
              r"\(\mathcal{R}_{\mathrm{cub}}\) & \(\mathcal{R}_{\mathrm{lw}}\) & defect")
    _write(_wrap("llllll", header, body), out / "exp7_momentum_table.tex")


def boundary_table(root: Path, out: Path) -> None:
    """Controlled domain sweep at fixed spacing (experiment 8, Section 5.9)."""
    rows = _read(root / "exp8_boundary_domain_sweep" / "boundary_domain_sweep.csv")
    rows.sort(key=lambda r: float(r["half_width"]))
    body = []
    for r in rows:
        err = float(r["err_to_largest_L2"])
        err_cell = "(reference)" if err == 0.0 else _sci(err)
        body.append(
            f"{_int(r['half_width'])} & {_int(r['M'])} & {_fixed(r['hx'],2)} & "
            f"{_sci(r['max_boundary_abs_u'])} & {err_cell} & "
            f"{_sci(r['max_RE_Jx'])} \\\\"
        )
    header = (r"\(L\) & \(M=N\) & \(h\) & max.\ outer-ring \(|U|\) & "
              r"\(\|U_L-U_{24}\|_2\) & \(\max\mathrm{RE}_{J_x}\)")
    _write(_wrap("llllll", header, body), out / "exp8_boundary_table.tex")


def performance_table(root: Path, out: Path) -> None:
    """Setup and per-step cost of the reference solver (experiment 9, Section 5.10)."""
    rows = _read(root / "exp9_performance_scaling" / "performance_scaling.csv")
    rows.sort(key=lambda r: int(r["Ntot"]))
    body = [
        f"{_int(r['M'])} & {_int(r['Ntot'])} & {_fixed(r['setup_seconds'],3)} & "
        f"{_fixed(r['per_step_ms'],1)} & {_int(r['mean_picard_iters'])} \\\\"
        for r in rows
    ]
    header = r"\(M=N\) & \(N_{\mathrm{tot}}\) & setup (s) & per step (ms) & Picard it."
    _write(_wrap("lllll", header, body), out / "exp9_performance_table.tex")


# ------------------------------------------------------------------------- driver

def write_all_tables(output_root: str | Path) -> None:
    """Regenerate every data table under ``<output_root>/tables/``."""
    root = Path(output_root)
    out = root / "tables"
    conservation_table(root, out)
    time_convergence_table(root, out)
    spatial_convergence_table(root, out)
    analytic_convergence_table(root, out)
    fractional_table(root, out)
    momentum_table(root, out)
    boundary_table(root, out)
    performance_table(root, out)
