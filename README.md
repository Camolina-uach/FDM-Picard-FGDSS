# Conservative Crank–Nicolson–Ortigueira–Picard solver for the FGDSS

Reference implementation and reproducibility package of the PhD thesis

> **Theoretical and Numerical Analysis of the Fractional Davey–Stewartson System**
> Carlos Alberto Molina Holguín
> Doctorado en Ciencia y Tecnología (Applied Mathematics), Centro Universitario
> de los Lagos, Universidad de Guadalajara, 2026.
> Advisors: Dr. Luis Armando Gallegos Infante and Dr. Jorge Eduardo Macías Díaz.

Version 1.1.0 is archived on Zenodo: <https://doi.org/10.5281/zenodo.23265326>.

The code solves the fractional generalized Davey–Stewartson system (FGDSS) with

- Ortigueira centered finite differences for the Riesz operators;
- homogeneous Dirichlet boundary conditions by zero extension;
- Crank–Nicolson midpoint time discretization;
- Picard iteration for the nonlinear short-wave equation, with the long-wave
  subsystem solved at the midpoint density inside every sweep;
- diagnostics for the discrete mass, the Hamiltonian, the momenta, and the
  long-wave residuals.

It produces every figure and data table of Chapter 5 of the thesis. Each module
starts with a docstring that maps its code to the sections, definitions, and
theorems of Chapter 4, so the implementation can be read side by side with the
thesis.

## Files

| File | Content |
| --- | --- |
| `fgdss_fd_operators.py` | Grid, Ortigueira coefficients, Riesz matrices, discrete inner products (Sections 4.1–4.4, 4.7). |
| `fgdss_solver.py` | Crank–Nicolson–Picard solver and invariant diagnostics (Sections 4.6–4.12). |
| `fgdss_experiments.py` | The nine experiments exp1–exp9 of Chapter 5. |
| `fgdss_plots.py` | Figures in the thesis style (Okabe–Ito palette, Computer Modern math). |
| `fgdss_tables.py` | LaTeX data tables (`outputs/tables/*.tex`) read by the thesis with `\input`. |
| `fgdss_babaoglu.py` | Qualitative sech/tanh profile of the classical degenerate GDS (not used as a benchmark). |
| `run_fgdss.py` | Entry point: runs exp1–exp9, then writes the figures and tables. |
| `exp10_manufactured_longwave.py` | Manufactured test of the long-wave recovery with `theta` active (Section 5.6). |
| `exp11_longwave_matrix_spectrum.py` | Smallest eigenvalue of `-M_h` for every grid, domain, and order of Chapter 5 (Section 5.1). |
| `exp12_fractional_order_profiles.py` | Conservation run repeated with common orders 2, 1.8, 1.5, 1.2 (Section 5.7). |
| `export_thesis_assets.py` | Copies figures and tables into the thesis folders. |
| `make_artifact_manifest.py` | Records the configuration of every experiment and SHA-256 hashes of code and results. |
| `tests/` | Regression tests (`test_fgdss.py`) and tests of the v4 experiments (`test_longwave_v4.py`). |
| `outputs_revision/` | The reviewed results used in the thesis, with `artifact_manifest.json`. |

## Requirements

Python 3.10 or newer with `numpy`, `scipy`, `matplotlib`, and `mpmath`
(the last one only for exp10):

```bash
pip install -r requirements.txt
```

The exact versions used for the archived results are recorded in
`requirements-lock.txt`: exp1–exp9 were run with Python 3.13.8, numpy 2.3.3,
scipy 1.16.2, and matplotlib 3.10.7; exp10–exp12 with Python 3.12.10,
numpy 2.5.2, scipy 1.18.1, matplotlib 3.11.1, and mpmath 1.3.0.

Run the tests (about half a minute) with:

```bash
python -m unittest discover -s tests -v
```

## Running the experiments

```bash
python run_fgdss.py
```

runs exp1–exp9 and writes CSV files, `summary.json` files, NPZ fields, figures
(`outputs/figures/`), and LaTeX tables (`outputs/tables/`) to `outputs/`. These
are the full thesis runs, not a smoke test; they take roughly 10–20 minutes,
depending on the hardware and the numerical libraries.

The experiments added in the final version of the thesis are separate scripts,
so the pipeline of `run_fgdss.py` is unchanged. They reuse the same grid,
operator, and solver modules:

```bash
python exp10_manufactured_longwave.py                # writes to outputs/
python exp11_longwave_matrix_spectrum.py             # a few minutes
python exp12_fractional_order_profiles.py run 2.0    # one order per call;
python exp12_fractional_order_profiles.py run 1.8    # each order s < 2 takes
python exp12_fractional_order_profiles.py run 1.5    # about 80 minutes
python exp12_fractional_order_profiles.py run 1.2
python exp12_fractional_order_profiles.py summarize  # statistics and figures
```

`fgdss_plots.py` also draws three figures of the conservation run from its
archived snapshots (cuts of `|U|` through the peak, contour maps of
`|U|, W, V, Q`, and cuts of `W, V, Q`) without rerunning exp2.

New runs write to `outputs/` and never overwrite `outputs_revision/`. To copy
figures and tables into the thesis folders (`figures/figures_chapter4/` and
`tables_chapter4/`):

```bash
python export_thesis_assets.py DEST                    # from outputs/
python export_thesis_assets.py DEST outputs_revision   # the archived results
```

## Experiments

| Folder | Purpose |
| --- | --- |
| `exp1_operator_verification` | Symmetry and negative semidefiniteness of the Riesz matrices. |
| `exp2_source_free_conservation` | Discrete mass and Hamiltonian conservation and momentum drift over time. |
| `exp3_time_convergence` | Temporal self-convergence: observed order in `dt` on a fixed grid. |
| `exp4_spatial_convergence` | Spatial self-convergence on nested grids: observed order in `h`. |
| `exp5_fractional_order_sensitivity` | Invariant defects across common fractional orders. |
| `exp6_integer_analytic_convergence` | Comparison with the exact linear Schrödinger standing wave at integer order. |
| `exp7_momentum_residual_balance` | Checks `dt Jz = R_disp + R_cub + R_lw` and reports the nonzero residuals. |
| `exp8_boundary_domain_sweep` | Domain sweep at fixed spacing with the same physical initial datum in every box. |
| `exp9_performance_scaling` | Setup and per-step wall time of the reference direct solver versus the number of unknowns. |
| `exp10_manufactured_longwave` | Manufactured long-wave recovery: second-order errors in `W, V, Q_h`, stagnation with `theta = 0` in the solver. |
| `exp11_longwave_matrix_spectrum` | Smallest eigenvalue of `-M_h` (positive in every configuration of Chapter 5). |
| `exp12_fractional_order_profiles` | Pulse shape and motion and the long-wave fields for each common order (the `s = 1.2` run reproduces exp2). |

## Archived results

`outputs_revision/` holds the results reported in the thesis, together with
`revision_metadata.json` and `artifact_manifest.json`. The manifest lists the
configuration of every experiment and the SHA-256 hash of every active file,
so the package can be checked byte by byte; snapshots kept only for provenance
are in `archive_excluded_snapshots/` and are hashed separately. The tests in
`tests/test_longwave_v4.py` regenerate selected results (including the first
40 steps of exp12 at `s = 2`) and compare them with the archive.

## Notes on interpretation

- The discrete Hamiltonian is built from the **full-order** quadratic forms
  `<-delta^(s) Z, Z>_h`, which are the exact invariants of the scheme. On the
  truncated Dirichlet grid they differ from the half-order seminorms
  `||delta^(s/2) Z||^2`.
- In converged runs the mass and the complete Hamiltonian are conserved to
  machine precision. The momenta are diagnostics, not exact invariants; exp7
  evaluates their three-term residual balance.
- The Picard iteration is unrelaxed (`picard_relaxation = 1`). Under-relaxation
  is rejected because it is no longer a mass-preserving Cayley update.
- The order of accuracy is measured by self-convergence (exp3, exp4), together
  with the exact standing wave at integer order (exp6) and the exact
  manufactured long-wave solution (exp10). The Babaoglu profile is not a
  solution of the Riesz-coupled FGDSS, because at `mu = 2` the coupling
  `delta^(mu/2)` is the Riesz operator of order 1, not a first derivative; it is
  therefore not used as a benchmark.

## Related software

The thesis cites two companion studies of the integer-order GDSS, whose codes
are archived separately:

- conservative Crank–Nicolson–Picard solver: <https://doi.org/10.5281/zenodo.23030677>;
- Fourier pseudo-spectral Strang-splitting solver: <https://doi.org/10.5281/zenodo.22891244>.

## License and citation

Released under the MIT License (see `LICENSE`). If you use this code, please
cite both the software (`CITATION.cff`) and the thesis.

- Archived version 1.1.0: <https://doi.org/10.5281/zenodo.23265326>
- Repository: <https://github.com/Camolina-uach/FDM-Picard-FGDSS>
