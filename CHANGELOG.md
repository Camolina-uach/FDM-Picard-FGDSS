# Changelog

## 1.1.0 — 2026-10-09 (final version of the thesis)

First public release, archived on Zenodo with DOI 10.5281/zenodo.23265326.
It matches the final version of the thesis
*Theoretical and Numerical Analysis of the Fractional Davey–Stewartson System*.

- New experiments: exp10 (manufactured test of the long-wave recovery with
  `theta` active), exp11 (smallest eigenvalue of `-M_h` for every configuration
  of Chapter 5), and exp12 (the conservation run repeated with common orders
  2, 1.8, 1.5, 1.2), with their archived results in `outputs_revision/`.
- New figures of the conservation run (cuts of `|U|`, contour maps, long-wave
  profiles) and of the order comparison (profiles, centroid, log-scale surfaces,
  maps and surfaces of `|U|, W, V`).
- New tests in `tests/test_longwave_v4.py`, including comparisons of
  regenerated results with the archive.
- Every source file has a header with the thesis title, the author, and the
  advisors, and every function has a docstring. No numerical code changed.
- `CITATION.cff`, `.zenodo.json`, and the SHA-256 manifest updated.

## 1.0.0 — 2026-07-31 (internal, not released)

Package of the earlier draft of the thesis: solver, experiments exp1–exp9,
figures, tables, and regression tests.
