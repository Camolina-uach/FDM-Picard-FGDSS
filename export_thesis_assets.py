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

"""Copy the generated figures and LaTeX tables into the thesis asset folders.

After ``python run_fgdss.py`` (which runs the suite and writes
``outputs/figures/`` and ``outputs/tables/``), run::

    python export_thesis_assets.py [DEST] [OUTPUT_ROOT]

to copy, into ``DEST``:

  * ``figures/figures_chapter4/`` -- the figures used by the experiments chapter,
    flattened and renamed to the ``expN_*`` names the chapter's
    ``\\includegraphics`` expects (each as PDF and PNG);
  * ``tables_chapter4/`` -- the LaTeX data tables (already correctly named).

``DEST`` defaults to ``~/Downloads``. ``OUTPUT_ROOT`` defaults to the working
``outputs/`` directory; pass ``outputs_revision`` to export the reviewed thesis
snapshot. For example::

    python export_thesis_assets.py ~/Documents/Thesis outputs_revision
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "outputs"

# Figures referenced by Chapter 5. The destination directory retains its legacy
# ``figures_chapter4`` name to match the existing thesis source tree.
# Source stem under outputs/figures/ -> destination stem.
# The map is explicit because two exp2 surfaces are renamed (they are not a plain
# flattening); keep it in sync with the chapter's \includegraphics if figures are
# added or renamed.
FIGURE_MAP = {
    "exp2/diagnostic_relative_drifts": "exp2_diagnostic_relative_drifts",
    "exp2/absU_surface3d": "exp2_amplitude3d",
    "exp2/fields_surface3d": "exp2_fields_surface3d",
    "exp3/time_convergence": "exp3_time_convergence",
    "exp4/spatial_convergence": "exp4_spatial_convergence",
    "exp5/fractional_order_drifts": "exp5_fractional_order_drifts",
    "exp6/integer_analytic_convergence": "exp6_integer_analytic_convergence",
    "exp7/momentum_residual_balance": "exp7_momentum_residual_balance",
    "exp8/boundary_domain_sweep": "exp8_boundary_domain_sweep",
    "exp9/performance_scaling": "exp9_performance_scaling",
    # thesis v4 additions
    "exp2/final_fields_contours": "exp2_final_fields_contours",
    "exp2/pulse_profiles": "exp2_pulse_profiles",
    "exp2/longwave_profiles": "exp2_longwave_profiles",
    "exp12/fractional_order_profiles": "exp12_fractional_order_profiles",
    "exp12/fractional_order_surfaces3d": "exp12_fractional_order_surfaces3d",
    "exp12/fractional_order_fields_maps": "exp12_fractional_order_fields_maps",
    "exp12/fractional_order_fields_surfaces3d": "exp12_fractional_order_fields_surfaces3d",
}


def export_assets(output_root: str | Path, dest_root: str | Path) -> None:
    """Copy the figures of ``FIGURE_MAP`` and every LaTeX table from
    ``output_root`` into the thesis folders under ``dest_root``.
    """
    output_root = Path(output_root)
    dest_root = Path(dest_root)
    figdst = dest_root / "figures" / "figures_chapter4"
    tabdst = dest_root / "tables_chapter4"
    figdst.mkdir(parents=True, exist_ok=True)
    tabdst.mkdir(parents=True, exist_ok=True)

    n_fig = 0
    missing = []
    for src_stem, dst_stem in FIGURE_MAP.items():
        found = False
        for ext in ("pdf", "png"):
            src = output_root / "figures" / f"{src_stem}.{ext}"
            if src.exists():
                shutil.copy2(src, figdst / f"{dst_stem}.{ext}")
                n_fig += 1
                found = True
        if not found:
            missing.append(src_stem)

    tables = sorted((output_root / "tables").glob("*.tex"))
    for t in tables:
        shutil.copy2(t, tabdst / t.name)

    print(f"Figures -> {figdst}  ({n_fig} files)")
    if missing:
        print(f"  WARNING: figures not found (run run_fgdss.py first): {missing}")
    print(f"Tables  -> {tabdst}  ({len(tables)} files)")
    if not tables:
        print("  WARNING: no tables found (run run_fgdss.py first).")


if __name__ == "__main__":
    dest = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "Downloads"
    source = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else OUTPUT_ROOT
    if not source.is_absolute():
        source = HERE / source
    export_assets(source, dest)
