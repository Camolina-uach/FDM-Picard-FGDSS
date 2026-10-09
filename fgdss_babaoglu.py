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

"""Babaoglu-type sech/tanh travelling profiles of the classical degenerate GDS.

This module provides closed-form travelling-wave profiles of the *classical*
degenerate generalized Davey--Stewartson (GDS) system for **qualitative**
visualisation only. It is deliberately kept out of the convergence and
conservation experiments, for a concrete reason: the FGDSS long-wave coupling
uses the Riesz operator ``delta^(mu/2)``, and at the integer value ``mu = 2``
that is the Riesz derivative of order 1 (Fourier symbol ``-|k|``), *not* the
first derivative ``partial_x`` appearing in the classical GDS coupling. Hence
the Babaoglu profile is not a solution of the Riesz-coupled FGDSS even at integer
orders, and cannot serve as an exact-solution convergence benchmark. The
rigorous convergence checks are the self-convergence studies (experiments 3-4)
and the exact linear Schrodinger benchmark (experiment 6).

The amplitude is additionally rescaled by ``amplitude_scale`` for numerically
gentle pictures, so these fields are not exact solutions of any discretised
system; use them only to illustrate the travelling-wave shape.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict
import json

import numpy as np

from fgdss_fd_operators import FDGrid

Array = np.ndarray


@dataclass
class BabaogluProfileParams:
    """Parameters of the qualitative sech/tanh travelling profile.

    ``k1, k2`` define the travelling coordinate, ``l1, l2, l3`` the phase,
    ``vcoef, j1, j2`` are the degenerate GDS coefficients, ``b0, c0`` are
    constant offsets of the long waves, and ``amplitude_scale`` rescales the
    amplitude for gentle pictures.
    """
    # Travelling coordinates and phase
    k1: float = 1.0
    k2: float = 0.55
    l1: float = 0.25
    l2: float = -0.10
    l3: float = -0.20
    d: float = 1.0

    # Degenerate GDS coefficients
    vcoef: float = 1.0
    j1: float = -1.0
    j2: float = -0.8

    # Optional offsets in the long waves
    b0: float = 0.0
    c0: float = 0.0

    # Manual amplitude scaling for numerically gentle tests
    amplitude_scale: float = 0.20

    def derived(self) -> Dict[str, float]:
        """Profile constants: inverse width ``phi0``, amplitude ``u0``, long-wave
        jumps ``bbar0, cbar0``, and travelling speed.
        """
        denom = self.j1 * self.k1**2 + self.j2 * self.k2**2 + self.vcoef * (self.k2**2 - self.k1**2)
        a1num = self.l1**2 + self.l2**2 - self.l3
        if a1num <= 0.0:
            raise ValueError("Need l1^2 + l2^2 - l3 > 0 for a localized sech profile.")
        phi0 = np.sqrt(a1num / (self.k1**2 + self.k2**2))
        # The closed form in the paper has sign-dependent existence conditions.
        # For a robust qualitative profile we take the positive absolute value
        # and apply amplitude_scale.  This keeps the shape while avoiding a
        # fragile parameter-selection step.
        raw_amp_sq = 2.0 * (self.k1**2 - self.k2**2) * a1num / denom if denom != 0 else np.nan
        u0 = self.amplitude_scale * np.sqrt(abs(raw_amp_sq)) if np.isfinite(raw_amp_sq) else self.amplitude_scale
        common = np.sqrt((self.k1**2 + self.k2**2) * a1num)
        bbar0 = 2.0 * self.j1 * self.k1 * common / denom if denom != 0 else 0.0
        cbar0 = 2.0 * self.j2 * self.k2 * common / denom if denom != 0 else 0.0
        speed = 2.0 * (self.d * self.k1 * self.l1 + self.k2 * self.l2)
        return {"phi0": float(phi0), "u0": float(u0), "bbar0": float(bbar0), "cbar0": float(cbar0), "speed": float(speed)}


def sech(z: Array) -> Array:
    """Hyperbolic secant."""
    return 1.0 / np.cosh(z)


def exact_fields(grid: FDGrid, t: float, bp: BabaogluProfileParams) -> Dict[str, Array]:
    """Evaluate the (rescaled, qualitative) profile on the interior grid at time ``t``.

    Returns ``U, W, V``, the classical first-derivative coupling
    ``Q_classical``, and ``abs_U``.
    """
    dct = bp.derived()
    X, Y = grid.X, grid.Y
    zeta = bp.k1 * X + bp.k2 * Y - dct["speed"] * t
    phase = bp.l1 * X + bp.l2 * Y - bp.l3 * t
    S = sech(dct["phi0"] * zeta)
    T = np.tanh(dct["phi0"] * zeta)
    U = -dct["u0"] * S * np.exp(1j * phase)
    W = bp.b0 - dct["bbar0"] * T
    V = bp.c0 - dct["cbar0"] * T
    # Classical first-derivative coupling used in the degenerate GDS profile.
    # d/dx tanh(phi0 zeta)=phi0*k1*sech^2(phi0 zeta), similarly for y.
    Wx = -dct["bbar0"] * dct["phi0"] * bp.k1 * S**2
    Vy = -dct["cbar0"] * dct["phi0"] * bp.k2 * S**2
    Q_classical = -(Wx + Vy)
    return {"U": U, "W": W, "V": V, "Q_classical": Q_classical, "abs_U": np.abs(U)}


def sample_babaoglu_profile(output_dir: str | Path, grid: FDGrid, times=(0.0, 0.5, 1.0), bp: BabaogluProfileParams | None = None) -> None:
    """Save exact Babaoglu-type profiles for comparison figures."""
    bp = BabaogluProfileParams() if bp is None else bp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for t in times:
        fields = exact_fields(grid, float(t), bp)
        np.savez_compressed(output_dir / f"babaoglu_exact_t_{t:.6f}.npz", x=grid.x, y=grid.y, t=t, **fields)
    with open(output_dir / "babaoglu_params.json", "w", encoding="utf-8") as f:
        json.dump({"profile": asdict(bp), "derived": bp.derived(), "grid": asdict(grid)}, f, indent=2)
