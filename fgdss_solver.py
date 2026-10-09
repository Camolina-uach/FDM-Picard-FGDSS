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

"""Crank--Nicolson -- Ortigueira -- Picard solver for the FGDSS (Chapter 4).

This module implements the fully discrete, source-free conservative scheme of
Chapter 4 and its invariant diagnostics. The short-wave equation is the midpoint
(Crank--Nicolson) discretisation

    i dt U + alpha Dx^lambda mu_t U + beta Dy^lambda mu_t U
      = (gamma mu_t rho + xi Q^{k+1/2}) mu_t U,                        (Section 4.6)

coupled to the long-wave elliptic subsystem, which is solved *at the midpoint
density* inside every Picard sweep -- the temporal centering that makes the
discrete energy identity close exactly (Section 4.9, Remark "the midpoint
centering inside the Picard loop is not redundant").

Map of the module
-----------------
* ``FGDSSParams``          -- model coefficients and fractional orders (Sec. 4.6).
* ``gaussian_initial_data``-- localized Dirichlet-compatible packet for the tests.
* ``FGDSSSolver.__init__`` -- assembles L_h and the long-wave block matrix M_h
                              (Section 4.7, "Matrix Formulation").
* ``solve_longwave_from_density`` / ``coupling_Q`` -- the elliptic solve and the
                              real long-wave coupling Q (Sections 4.6, 4.9).
* ``picard_step``          -- one time step by Picard fixed-point iteration
                              (Section 4.9); mass is conserved exactly at every
                              sweep (Cayley transform), energy after convergence.
* ``mass`` / ``energy`` / ``momentum`` -- the discrete invariants of Section 4.12.
                              Energy uses the *full-order* quadratic forms
                              <-delta^(s) Z, Z>_h (the exact discrete invariant),
                              not the half-order seminorms.
* ``momentum_residual_terms`` -- the R_disp / R_cub / R_lw decomposition of the
                              momentum balance (Section 4.12, Proposition
                              "Residual form of the discrete momentum balance").
* ``run`` + CSV/NPZ helpers -- driver and output for the experiment suite.

Only interior unknowns are stored; homogeneous Dirichlet values are zero by the
zero-extension convention of Section 4.3.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math
import time
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve

from fgdss_fd_operators import (
    Array,
    FDGrid,
    Riesz2DCache,
    vec,
    mat,
    inner_h,
    l2_norm_h,
    l4_power_h,
    centered_first_x,
    centered_first_y,
)


# =============================================================================
# Model parameters (Section 4.6): coefficients and fractional orders
# =============================================================================

@dataclass
class FGDSSParams:
    """FGDSS coefficients, fractional orders, and solver options.

    The coefficient names match the chapter: ``alpha, beta`` scale the short-wave
    dispersion, ``gamma`` the cubic self-interaction, ``xi`` the long-wave
    coupling, and ``psi, eta, phi, chi, theta`` the long-wave elliptic block
    ``M_h``. The fractional orders ``lam, mu, nu, kappa, rho, sigma`` are the
    Riesz orders of the corresponding operators and must lie in ``(1, 2]``.

    ``theta`` may be given explicitly; if left as ``None`` it defaults to the
    structural value ``sqrt((phi-psi)(eta-chi))`` (real only when the product is
    non-negative), a convenient closed form that keeps the long-wave block
    matrix well posed for the default coefficient set. Lower orders remain
    available internally to construct the required half-order operators.
    """

    # Short-wave coefficients
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    xi: float = 1.0

    # Long-wave coefficients
    psi: float = 1.0
    eta: float = 2.0
    phi: float = 2.0
    chi: float = 1.0
    theta: Optional[float] = None

    # Fractional (Riesz) model orders, each in (1, 2]
    lam: float = 1.2
    mu: float = 1.2
    nu: float = 1.2
    kappa: float = 1.2
    rho: float = 1.2
    sigma: float = 1.2

    # Time discretization
    dt: float = 5.0e-4
    t_final: float = 5.0e-2

    # Picard fixed-point options (Section 4.9)
    picard_tol: float = 1.0e-10
    picard_max_iter: int = 30
    # The conservative reference algorithm uses the unrelaxed Cayley update.
    picard_relaxation: float = 1.0

    # Final/reproducible runs fail rather than silently accepting an invalid step.
    fail_on_nonconvergence: bool = True

    def __post_init__(self) -> None:
        """Fill in the structural ``theta`` if needed and validate the parameters.

        Raises ``ValueError`` for orders outside ``(1, 2]``, a non-positive time
        step or tolerance, and any Picard relaxation other than 1.
        """
        # Structural closed form for theta (see class docstring).
        if self.theta is None:
            val = (self.phi - self.psi) * (self.eta - self.chi)
            if val < 0.0:
                raise ValueError("theta=None requires (phi-psi)(eta-chi) >= 0.")
            self.theta = float(math.sqrt(val))
        for name in ("lam", "mu", "nu", "kappa", "rho", "sigma"):
            val = getattr(self, name)
            if not (1.0 < val <= 2.0):
                raise ValueError(f"{name} must be in (1,2] for the thesis model.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.t_final < 0.0:
            raise ValueError("t_final must be non-negative.")
        if self.picard_tol <= 0.0:
            raise ValueError("picard_tol must be positive.")
        if not isinstance(self.picard_max_iter, int) or isinstance(self.picard_max_iter, bool) or self.picard_max_iter < 1:
            raise ValueError("picard_max_iter must be a positive integer.")
        if self.picard_relaxation != 1.0:
            raise ValueError(
                "The structure-preserving reference solver requires "
                "picard_relaxation=1. Under-relaxation is not a Cayley update "
                "and does not preserve mass at each sweep."
            )


# =============================================================================
# Initial data
# =============================================================================

def gaussian_initial_data(
    grid: FDGrid,
    amplitude: float = 0.25,
    sigma_x: float = 4.5,
    sigma_y: float = 3.0,
    x0: float = 0.0,
    y0: float = 0.0,
    kx0: float = 0.35,
    ky0: float = 0.20,
    use_window: bool = True,
) -> Array:
    """Localized, Dirichlet-compatible Gaussian wave packet.

    A Gaussian envelope with a linear phase ``exp(i(kx0 x + ky0 y))`` (so the
    packet carries non-zero momentum). When ``use_window`` is set, the envelope
    is multiplied by a ``sin^2`` window that vanishes on the boundary, making the
    data compatible with the homogeneous Dirichlet treatment of Section 4.3 and
    keeping the boundary amplitude small for the conservation experiments.
    """
    X, Y = grid.X, grid.Y
    amp = amplitude * np.exp(-((X - x0) ** 2 / (2 * sigma_x**2) + (Y - y0) ** 2 / (2 * sigma_y**2)))
    if use_window:
        Wx = np.sin(np.pi * (X - grid.xL) / (grid.xR - grid.xL)) ** 2
        Wy = np.sin(np.pi * (Y - grid.yL) / (grid.yR - grid.yL)) ** 2
        amp = amp * Wx * Wy
    phase = np.exp(1j * (kx0 * X + ky0 * Y))
    return amp * phase


# =============================================================================
# The solver (Sections 4.6 - 4.12)
# =============================================================================


class FGDSSSolver:
    """Reference sparse implementation of the FGDSS CN--Picard scheme."""

    def __init__(self, grid: FDGrid, params: FGDSSParams):
        """Assemble the operators of the matrix formulation (Section 4.7).

        Builds the short-wave dispersive operator ``L_h = alpha D_x^lam + beta
        D_y^lam`` and the symmetric long-wave block matrix

            M_h = [[psi D_x^kappa + eta D_y^kappa,   theta D_xy^sigma        ],
                   [theta D_xy^sigma,                phi D_x^rho + chi D_y^rho]],

        which is factorised once (``splu``) and reused for every elliptic solve.
        """
        self.grid = grid
        self.p = params
        self.ops = Riesz2DCache(grid)
        self.I = sparse.eye(grid.ntot, format="csc", dtype=complex)

        # Short-wave linear dispersive operator L_h (real, symmetric).
        self.Lu = (params.alpha * self.ops.Dx(params.lam) + params.beta * self.ops.Dy(params.lam)).tocsc()

        # Long-wave block matrix M_h and its LU factorization, computed once.
        Aww = (params.psi * self.ops.Dx(params.kappa) + params.eta * self.ops.Dy(params.kappa)).tocsc()
        Avv = (params.phi * self.ops.Dx(params.rho) + params.chi * self.ops.Dy(params.rho)).tocsc()
        Awv = (params.theta * self.ops.Dxy(params.sigma)).tocsc()
        self.Mlong = sparse.bmat([[Aww, Awv], [Awv, Avv]], format="csc")
        try:
            self._long_lu = splu(self.Mlong)
        except RuntimeError as exc:
            raise RuntimeError(
                "The long-wave block matrix could not be factorized. Check coefficient signs, fractional orders, and theta."
            ) from exc

    # ------------------------------------------------------------------ long-wave
    # Elliptic subsystem (Section 4.6): given a density rho, solve M_h (W, V)^T =
    # (D_x^{mu/2} rho, D_y^{nu/2} rho)^T for the long-wave potentials, and form
    # the real coupling Q = D_x^{mu/2} W + D_y^{nu/2} V used in the short-wave eq.
    def solve_longwave_from_density(self, rho: Array) -> Tuple[Array, Array]:
        """Solve the long-wave elliptic subsystem for a given short-wave density."""
        rhs_w = self.ops.Dx(self.p.mu / 2.0) @ vec(rho)
        rhs_v = self.ops.Dy(self.p.nu / 2.0) @ vec(rho)
        sol = self._long_lu.solve(np.concatenate([rhs_w, rhs_v]))
        n = self.grid.ntot
        W = mat(sol[:n], self.grid).real
        V = mat(sol[n:], self.grid).real
        return W, V

    def coupling_Q(self, W: Array, V: Array) -> Array:
        """Real long-wave coupling ``Q = D_x^(mu/2) W + D_y^(nu/2) V``.

        This is the potential that multiplies ``U`` in the short-wave equation.
        """
        Q = self.ops.apply_x(self.p.mu / 2.0, W) + self.ops.apply_y(self.p.nu / 2.0, V)
        return Q.real

    def initialize_longwave(self, U0: Array) -> Tuple[Array, Array]:
        """Long-wave fields compatible with ``U0``: solve the elliptic subsystem
        with the density ``|U0|^2``.
        """
        return self.solve_longwave_from_density(np.abs(U0) ** 2)

    # ------------------------------------------------------------- Picard time step
    # One step t_k -> t_{k+1} by the Picard fixed-point iteration of Section 4.9.
    # Each sweep freezes the nonlinear potential at the current iterate, forms the
    # midpoint density, re-solves the long-wave subsystem, and solves one linear
    # Crank--Nicolson system; the long-wave fields are recovered afterwards from
    # the converged midpoint density.
    def picard_step(
        self,
        U: Array,
        W: Array,
        V: Array,
        return_info: bool = False,
    ) -> Tuple[Array, Array, Array, Dict[str, float]]:
        """Advance one time step by Picard iteration.

        At every Picard sweep, rho^{k+1/2,(r)} is formed from the current
        candidate U^{k+1,(r)} and the long-wave subsystem is solved again.

        Returns ``(U, W, V, info)`` at the new time level, where ``info`` holds
        the number of sweeps, the convergence flag, and the last relative
        change. ``return_info`` is kept for compatibility; ``info`` is always
        returned.
        """
        p, g = self.p, self.grid
        dt = p.dt
        U_old = np.asarray(U, dtype=complex)
        U_iter = U_old.copy()
        rho_old = np.abs(U_old) ** 2
        U_old_vec = vec(U_old)

        converged = False
        last_rel_change = np.inf

        for it in range(1, p.picard_max_iter + 1):
            rho_new_iter = np.abs(U_iter) ** 2
            rho_mid = 0.5 * (rho_old + rho_new_iter)
            W_mid, V_mid = self.solve_longwave_from_density(rho_mid)
            Q_mid = self.coupling_Q(W_mid, V_mid)
            nonlinear_potential = p.gamma * rho_mid + p.xi * Q_mid

            # Linear Crank--Nicolson solve with the frozen (real) potential:
            #   [iI + (dt/2)(L_h - N)] U^{new} = [iI - (dt/2)(L_h - N)] U^{old}.
            # Because L_h - N is real symmetric, this is a Cayley transform of a
            # Hermitian operator, hence unitary: mass is conserved exactly at
            # every sweep, independently of Picard convergence (Section 4.12).
            Dpot = sparse.diags(vec(nonlinear_potential), offsets=0, format="csc", dtype=complex)
            A = (1j * self.I + 0.5 * dt * self.Lu - 0.5 * dt * Dpot).tocsc()
            b = (1j * U_old_vec - 0.5 * dt * (self.Lu @ U_old_vec) + 0.5 * dt * (Dpot @ U_old_vec))
            U_candidate = mat(spsolve(A, b), g)

            denom = 1.0 + l2_norm_h(U_candidate, g)
            last_rel_change = l2_norm_h(U_candidate - U_iter, g) / denom
            U_iter = U_candidate
            if last_rel_change <= p.picard_tol:
                converged = True
                break

        if not converged:
            msg = (
                f"Picard failed to converge after {p.picard_max_iter} iterations; "
                f"last relative change={last_rel_change:.3e}."
            )
            if p.fail_on_nonconvergence:
                raise RuntimeError(msg)
            warnings.warn(msg, RuntimeWarning)

        # Recompute midpoint fields from the converged U, then recover integer W,V.
        rho_mid = 0.5 * (rho_old + np.abs(U_iter) ** 2)
        W_mid, V_mid = self.solve_longwave_from_density(rho_mid)
        W_new = 2.0 * W_mid - W
        V_new = 2.0 * V_mid - V
        info = {
            "picard_iters": float(it),
            "picard_converged": float(converged),
            "picard_rel_change": float(last_rel_change),
        }
        return U_iter, W_new.real, V_new.real, info

    # ---------------------------------------------------------- discrete invariants
    # The discrete mass M_h and Hamiltonian E_{theta,h} of Section 4.12; the
    # momenta J_{x,h}, J_{y,h} are diagnostics, not exact invariants.
    def mass(self, U: Array) -> float:
        """Discrete mass ``M_h = ||U||_2^2`` (conserved to machine precision)."""
        return float(self.grid.hx * self.grid.hy * np.sum(np.abs(U) ** 2))

    def energy(self, U: Array, W: Array, V: Array) -> float:
        """Discrete Hamiltonian ``E_{theta,h}`` of Section 4.12.

            E = alpha <-D_x^lam U, U> + beta <-D_y^lam U, U> + (gamma/2) ||U||_4^4
                - (xi/2) [psi <-D_x^kappa W, W> + eta <-D_y^kappa W, W>
                          + phi <-D_x^rho V, V> + chi <-D_y^rho V, V>]
                + xi theta <D_xy^sigma V, W>,

        with the discrete inner product ``<., .>_h``.
        """
        # The discrete Hamiltonian is built from the *full-order* quadratic forms
        # <-delta_z^(s) Z, Z>_h, which are the quantities the CN--Ortigueira step
        # conserves exactly.  On the truncated Dirichlet grid these are NOT equal
        # to the half-order seminorms ||delta_z^(s/2) Z||^2 (the semigroup
        # identity (delta^(s/2))^2 = -delta^(s) only holds on the periodic/
        # infinite grid), so using the seminorms would measure a different
        # finite-grid quadratic form and would not satisfy the exact identity.
        p, g, ops = self.p, self.grid, self.ops

        def qform_x(order: float, Z: Array) -> float:
            """Full-order quadratic form ``<-D_x^(order) Z, Z>_h``."""
            return float(np.real(inner_h(-ops.apply_x(order, Z), Z, g)))

        def qform_y(order: float, Z: Array) -> float:
            """Full-order quadratic form ``<-D_y^(order) Z, Z>_h``."""
            return float(np.real(inner_h(-ops.apply_y(order, Z), Z, g)))

        ctheta = np.real(inner_h(ops.apply_xy(p.sigma, V), W, g))
        val = (
            p.alpha * qform_x(p.lam, U)
            + p.beta * qform_y(p.lam, U)
            + 0.5 * p.gamma * l4_power_h(U, g)
            - 0.5
            * p.xi
            * (
                p.psi * qform_x(p.kappa, W)
                + p.eta * qform_y(p.kappa, W)
                + p.phi * qform_x(p.rho, V)
                + p.chi * qform_y(p.rho, V)
            )
            + p.xi * p.theta * ctheta
        )
        return float(np.real(val))

    def momentum(self, U: Array) -> Tuple[float, float]:
        """Discrete momenta ``J_{z,h} = -2 Im <nabla_z U, U>_h`` for z = x, y.

        They use the classical centered first differences and are diagnostics,
        not exact invariants of the scheme (Section 4.12).
        """
        ux = centered_first_x(U, self.grid)
        uy = centered_first_y(U, self.grid)
        Jx = -2.0 * self.grid.hx * self.grid.hy * np.sum(np.imag(ux * np.conj(U)))
        Jy = -2.0 * self.grid.hx * self.grid.hy * np.sum(np.imag(uy * np.conj(U)))
        return float(Jx), float(Jy)

    # ------------------------------------------------- momentum residual balance
    def _apply_L(self, Z: Array) -> Array:
        """Linear dispersive operator L_h = alpha Dx^lam + beta Dy^lam."""
        return mat(self.Lu @ vec(Z), self.grid)

    @staticmethod
    def _grad(Z: Array, grid: FDGrid, axis: str) -> Array:
        """Centered first difference of ``Z`` along ``axis`` (``"x"`` or ``"y"``)."""
        return centered_first_x(Z, grid) if axis == "x" else centered_first_y(Z, grid)

    def _product_defect(self, A: Array, G: Array, axis: str) -> Array:
        """Discrete product-rule defect Pi_z(A,G)=grad_z(AG)-A grad_z G-(grad_z A)G."""
        g = self.grid
        return self._grad(A * G, g, axis) - A * self._grad(G, g, axis) - self._grad(A, g, axis) * G

    def momentum_residual_terms(
        self,
        U_old: Array,
        U_new: Array,
        W_old: Array,
        W_new: Array,
        V_old: Array,
        V_new: Array,
    ) -> Dict[str, float]:
        """Evaluate the discrete momentum-balance residuals of Section 4.12
        (Proposition "Residual form of the discrete momentum balance").

        Returns, for z in {x,y}, the dispersive, cubic and long-wave residuals
        R_disp, R_cub, R_lw and the measured defect dt Jz - (R_disp+R_cub+R_lw),
        which vanishes to solver tolerance when the scheme identity holds.
        """
        p, g = self.p, self.grid
        G = 0.5 * (np.asarray(U_old, dtype=complex) + np.asarray(U_new, dtype=complex))
        R = 0.5 * (np.abs(U_old) ** 2 + np.abs(U_new) ** 2)            # mu_t rho
        W_mid = 0.5 * (W_old + W_new)
        V_mid = 0.5 * (V_old + V_new)
        Q = self.coupling_Q(W_mid, V_mid)                              # real potential
        LG = self._apply_L(G)
        absG2 = np.abs(G) ** 2
        J_old = self.momentum(U_old)
        J_new = self.momentum(U_new)
        out: Dict[str, float] = {}
        for ax_idx, axis in enumerate(("x", "y")):
            gradG = self._grad(G, g, axis)
            # dispersive commutator residual
            r_disp = 2.0 * np.real(
                inner_h(gradG, LG, g) - inner_h(self._grad(LG, g, axis), G, g)
            )

            def _pot(A: Array) -> float:
                """Residual contributed by a real potential ``A`` (the cubic density or ``Q``)."""
                return float(
                    2.0 * np.real(inner_h(self._grad(A, g, axis), absG2, g))
                    + 2.0 * np.real(inner_h(self._product_defect(A, G, axis), G, g))
                )

            r_cub = p.gamma * _pot(R)
            r_lw = p.xi * _pot(Q)
            Jz_old = J_old[ax_idx]
            Jz_new = J_new[ax_idx]
            dt_Jz = (Jz_new - Jz_old) / p.dt
            out[f"R_disp_{axis}"] = float(r_disp)
            out[f"R_cub_{axis}"] = float(r_cub)
            out[f"R_lw_{axis}"] = float(r_lw)
            out[f"dtJ_{axis}"] = float(dt_Jz)
            out[f"balance_defect_{axis}"] = float(dt_Jz - (r_disp + r_cub + r_lw))
        return out

    def longwave_residuals(self, U: Array, W: Array, V: Array) -> Tuple[float, float]:
        """Discrete L^2 norms of the residuals of the two long-wave equations.

        They measure how well ``(W, V)`` satisfy the elliptic subsystem for the
        density ``|U|^2``; at a recovered time level they are at round-off.
        """
        p, ops, g = self.p, self.ops, self.grid
        rho = np.abs(U) ** 2
        rw = (
            p.psi * ops.apply_x(p.kappa, W)
            + p.eta * ops.apply_y(p.kappa, W)
            + p.theta * ops.apply_xy(p.sigma, V)
            - ops.apply_x(p.mu / 2.0, rho)
        )
        rv = (
            p.phi * ops.apply_x(p.rho, V)
            + p.chi * ops.apply_y(p.rho, V)
            + p.theta * ops.apply_xy(p.sigma, W)
            - ops.apply_y(p.nu / 2.0, rho)
        )
        return l2_norm_h(rw, g), l2_norm_h(rv, g)

    def boundary_amplitude(self, U: Array) -> float:
        """Maximum amplitude on the outermost *interior* ring of nodes.

        The boundary nodes themselves are identically zero under the homogeneous
        Dirichlet treatment, so the relevant boundary-contamination diagnostic is
        the amplitude at the interior nodes adjacent to the boundary.
        """
        U = np.asarray(U)
        if U.shape[0] < 1 or U.shape[1] < 1:
            return 0.0
        ring = np.concatenate([U[0, :], U[-1, :], U[:, 0], U[:, -1]])
        return float(np.max(np.abs(ring)))

    def diagnostics_row(
        self,
        n: int,
        t: float,
        U: Array,
        W: Array,
        V: Array,
        info: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """One row of the diagnostic history at step ``n`` and time ``t``.

        Contains the mass, the Hamiltonian, both momenta, the long-wave residuals,
        the boundary-ring amplitude, and, if given, the Picard information.
        """
        Jx, Jy = self.momentum(U)
        rw, rv = self.longwave_residuals(U, W, V)
        row = {
            "n": int(n),
            "t": float(t),
            "M": self.mass(U),
            "E_theta": self.energy(U, W, V),
            "Jx": Jx,
            "Jy": Jy,
            "res_W": rw,
            "res_V": rv,
            "boundary_max_abs_u": self.boundary_amplitude(U),
        }
        if info:
            row.update(info)
        return row

    # --------------------------------------------------------------- time-stepping
    def run(
        self,
        U0: Array,
        W0: Optional[Array] = None,
        V0: Optional[Array] = None,
        save_every: int = 10,
        snapshot_times: Optional[List[float]] = None,
    ) -> Dict[str, object]:
        """Integrate from ``U0`` to ``t_final`` and return history and snapshots.

        If ``W0, V0`` are omitted they are obtained by solving the elliptic
        subsystem from ``|U0|^2`` (compatible seed data). Diagnostics are recorded
        every ``save_every`` steps; full fields are stored at ``snapshot_times``.
        """
        if not isinstance(save_every, int) or isinstance(save_every, bool) or save_every < 1:
            raise ValueError("save_every must be a positive integer.")
        if W0 is None or V0 is None:
            W0, V0 = self.initialize_longwave(U0)
        else:
            # The discrete energy identity relies on the long-wave fields being
            # compatible with the elliptic subsystem at t_0 (recovery lemma). If
            # inconsistent seed data are supplied, discrete energy conservation
            # silently degrades, so warn rather than fail.
            rw0, rv0 = self.longwave_residuals(U0, W0, V0)
            scale = 1.0 + l2_norm_h(np.abs(U0) ** 2, self.grid)
            if max(rw0, rv0) > 1e-6 * scale:
                warnings.warn(
                    "Initial W0,V0 are not compatible with the discrete elliptic "
                    f"subsystem (residuals {rw0:.2e}, {rv0:.2e}); discrete energy "
                    "conservation is only guaranteed for compatible long-wave data.",
                    RuntimeWarning,
                )
        U, W, V = U0.copy(), W0.copy(), V0.copy()
        Nt = int(round(self.p.t_final / self.p.dt))
        if not np.isclose(Nt * self.p.dt, self.p.t_final):
            raise ValueError("t_final must be an integer multiple of dt for this run helper.")

        snapshot_times = [] if snapshot_times is None else list(snapshot_times)
        snapshot_steps = {int(round(ts / self.p.dt)): ts for ts in snapshot_times if 0 <= ts <= self.p.t_final}
        history: List[Dict[str, float]] = []
        picard_history: List[Dict[str, float]] = []
        snapshots: Dict[str, Dict[str, Array]] = {}
        t0_wall = time.perf_counter()

        row0 = self.diagnostics_row(0, 0.0, U, W, V)
        history.append(row0)
        if 0 in snapshot_steps:
            snapshots[f"t_{0.0:.6f}"] = {"U": U.copy(), "W": W.copy(), "V": V.copy(), "Q": self.coupling_Q(W, V)}

        for n in range(Nt):
            U, W, V, info = self.picard_step(U, W, V, return_info=True)
            t = (n + 1) * self.p.dt
            picard_history.append({"n": float(n + 1), "t": float(t), **info})
            if (n + 1) % save_every == 0 or (n + 1) == Nt:
                history.append(self.diagnostics_row(n + 1, t, U, W, V, info))
            if (n + 1) in snapshot_steps:
                snapshots[f"t_{t:.6f}"] = {"U": U.copy(), "W": W.copy(), "V": V.copy(), "Q": self.coupling_Q(W, V)}

        wall_seconds = time.perf_counter() - t0_wall
        return {
            "grid": self.grid,
            "params": self.p,
            "history": history,
            "picard_history": picard_history,
            "snapshots": snapshots,
            "final": {"U": U, "W": W, "V": V, "Q": self.coupling_Q(W, V)},
            "wall_seconds": wall_seconds,
        }


# =============================================================================
# Output helpers: relative drifts, CSV history, NPZ fields (used by the suite)
# =============================================================================

def add_relative_drifts(history: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Append absolute and relative drifts ``RE`` of the invariants to each row.

    The relative drift ``RE_X = |X^k - X^0| / (1 + |X^0|)`` is the quantity
    reported in the numerical experiments (Section 4.12).
    """
    if not history:
        return history
    first = history[0]
    keys = ["M", "E_theta", "Jx", "Jy"]
    out = []
    for row in history:
        r = dict(row)
        for k in keys:
            base = float(first[k])
            val = float(row[k])
            ad = abs(val - base)
            re = ad / (1.0 + abs(base))
            r[f"AD_{k}"] = ad
            r[f"RE_{k}"] = re
        out.append(r)
    return out


def write_history_csv(history: List[Dict[str, float]], path: str | Path) -> None:
    """Write diagnostic history even when later rows contain extra fields.

    The initial diagnostic row is produced before the first Picard solve, so it
    may not contain Picard metadata such as ``picard_iters``.  CSV fieldnames
    must therefore be built from the union of all row keys, not only from the
    first row.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return

    preferred = [
        "n", "t",
        "M", "AD_M", "RE_M",
        "E_theta", "AD_E_theta", "RE_E_theta",
        "Jx", "AD_Jx", "RE_Jx",
        "Jy", "AD_Jy", "RE_Jy",
        "res_W", "res_V",
        "picard_iters", "picard_converged", "picard_rel_change",
    ]
    all_keys = set().union(*(row.keys() for row in history))
    keys = [k for k in preferred if k in all_keys]
    keys += sorted(k for k in all_keys if k not in keys)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in keys})


def save_run_npz(result: Dict[str, object], outdir: str | Path) -> None:
    """Write the fields of a solver run to ``outdir``.

    Creates ``final_fields.npz``, one ``snapshots/t_<time>.npz`` per snapshot
    (with a ``snapshot_manifest.json``), and ``params.json`` with the grid,
    the parameters, and the wall time. Each NPZ file stores ``x, y, U,
    abs_U, W, V, Q``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    grid: FDGrid = result["grid"]  # type: ignore[assignment]
    params: FGDSSParams = result["params"]  # type: ignore[assignment]
    final = result["final"]  # type: ignore[assignment]
    np.savez_compressed(
        outdir / "final_fields.npz",
        x=grid.x,
        y=grid.y,
        U=final["U"],
        abs_U=np.abs(final["U"]),
        W=final["W"],
        V=final["V"],
        Q=final["Q"],
    )
    # Save snapshots as one npz per time to avoid ragged object arrays.
    snapdir = outdir / "snapshots"
    snapdir.mkdir(exist_ok=True)
    snapshots = result["snapshots"]  # type: ignore[assignment]
    snapshot_names = sorted(str(name) for name in snapshots)
    expected_files = {f"{name}.npz" for name in snapshot_names}
    # This directory is generated output: remove only obsolete files that match
    # the solver's own snapshot naming convention, so runs cannot be mixed.
    for stale in snapdir.glob("t_*.npz"):
        if stale.name not in expected_files:
            stale.unlink()
    for name in snapshot_names:
        fields = snapshots[name]
        np.savez_compressed(
            snapdir / f"{name}.npz",
            x=grid.x,
            y=grid.y,
            U=fields["U"],
            abs_U=np.abs(fields["U"]),
            W=fields["W"],
            V=fields["V"],
            Q=fields["Q"],
        )
    with open(snapdir / "snapshot_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"files": sorted(expected_files)}, f, indent=2)
    with open(outdir / "params.json", "w", encoding="utf-8") as f:
        json.dump({"grid": asdict(grid), "params": asdict(params), "wall_seconds": result["wall_seconds"]}, f, indent=2)
