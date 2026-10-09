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

"""Discrete Riesz operators, grid, and inner products for the FGDSS solver.

This module implements the spatial building blocks of the Crank--Nicolson --
Ortigueira -- Picard scheme analyzed in Chapter 4 of the thesis
("A Conservative Crank--Nicolson Finite-Difference Scheme for the FGDSS").

Correspondence with the chapter
-------------------------------
* ``ortigueira_coefficients`` -- the centered fractional coefficients
  ``g_j^(s)`` of Definition "Ortigueira fractional centered differences"
  (Section 4.1).
* ``riesz_1d`` / ``Riesz2DCache`` -- the discrete Riesz operators

      delta_x^(s) Z_{m,n} = -h_x^{-s} sum_j g_{m-j}^(s) Z_{j,n},

  in the negative-semidefinite sign convention of Section 4.1, assembled as
  symmetric Toeplitz matrices and combined into 2-D operators through the
  Kronecker products of Section 4.7 (Matrix Formulation).
* ``inner_h`` / ``l2_norm_h`` / ``l4_power_h`` -- the discrete inner product and
  norms of Section 4.4 (Discrete Inner Product and Symmetry Properties).
* ``centered_first_x`` / ``centered_first_y`` -- the classical centered
  differences ``nabla_x, nabla_y`` used *only* for the momentum diagnostics of
  Section 4.12; deliberately distinct from the fractional operators.

Conventions
-----------
Only interior grid values are stored (homogeneous Dirichlet boundary values are
identically zero and enforced by *zero extension*, Section 4.3). Interior fields
are shaped ``(nx, ny)`` and vectorised with x running fastest (Fortran order),
which is what the Kronecker layout ``kron(I_y, D_x)`` / ``kron(D_y, I_x)``
assumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict

import numpy as np
from scipy import sparse
from scipy.special import gamma
from scipy.linalg import toeplitz

Array = np.ndarray


# =============================================================================
# Computational grid (Section 4.2, "Computational Grid and Notation")
# =============================================================================


@dataclass(frozen=True)
class FDGrid:
    """Interior finite-difference grid on [xL,xR] x [yL,yR].

    M,N are the numbers of subintervals in x,y.  The number of interior
    unknowns is nx=M-1 and ny=N-1.
    """
    xL: float = -20.0
    xR: float = 20.0
    yL: float = -20.0
    yR: float = 20.0
    M: int = 64
    N: int = 64

    def __post_init__(self) -> None:
        """Check that interior nodes exist and that the rectangle is valid."""
        if self.M < 3 or self.N < 3:
            raise ValueError("M and N must be at least 3 so that interior nodes exist.")
        if not (self.xL < self.xR and self.yL < self.yR):
            raise ValueError("Invalid rectangular domain.")

    @property
    def hx(self) -> float:
        """Mesh width in x."""
        return (self.xR - self.xL) / self.M

    @property
    def hy(self) -> float:
        """Mesh width in y."""
        return (self.yR - self.yL) / self.N

    @property
    def nx(self) -> int:
        """Number of interior nodes in x."""
        return self.M - 1

    @property
    def ny(self) -> int:
        """Number of interior nodes in y."""
        return self.N - 1

    @property
    def ntot(self) -> int:
        """Total number of interior unknowns, ``nx * ny``."""
        return self.nx * self.ny

    @property
    def x_full(self) -> Array:
        """All ``M + 1`` nodes in x, boundary nodes included."""
        return np.linspace(self.xL, self.xR, self.M + 1)

    @property
    def y_full(self) -> Array:
        """All ``N + 1`` nodes in y, boundary nodes included."""
        return np.linspace(self.yL, self.yR, self.N + 1)

    @property
    def x(self) -> Array:
        """Interior nodes in x."""
        return self.x_full[1:-1]

    @property
    def y(self) -> Array:
        """Interior nodes in y."""
        return self.y_full[1:-1]

    @property
    def X(self) -> Array:
        """x coordinate of every interior node, shape ``(nx, ny)``."""
        X, _ = np.meshgrid(self.x, self.y, indexing="ij")
        return X

    @property
    def Y(self) -> Array:
        """y coordinate of every interior node, shape ``(nx, ny)``."""
        _, Y = np.meshgrid(self.x, self.y, indexing="ij")
        return Y


# -----------------------------------------------------------------------------
# Grid <-> vector helpers.
#
# The lexicographic (x-fastest) ordering used here is the one assumed by the
# Kronecker products of Section 4.7: for an interior field of shape (nx, ny),
# vec() flattens with x running fastest so that kron(I_y, D_x) applies D_x within
# each y-column and kron(D_y, I_x) applies D_y across columns.
# -----------------------------------------------------------------------------

def vec(Z: Array) -> Array:
    """Flatten an interior field ``(nx, ny)`` to a vector, x running fastest."""
    return np.asarray(Z).reshape(-1, order="F")


def mat(z: Array, grid: FDGrid) -> Array:
    """Inverse of :func:`vec`: reshape a vector back to ``(nx, ny)``."""
    return np.asarray(z).reshape((grid.nx, grid.ny), order="F")


def full_with_zero_boundary(Z: Array, grid: FDGrid) -> Array:
    """Embed an interior field into the full grid with zero Dirichlet boundary.

    This realises the zero-extension interpretation of Section 4.3: values
    outside the interior are set to zero, which is what the fractional sums and
    the centered momentum differences assume near the boundary.
    """
    out = np.zeros((grid.M + 1, grid.N + 1), dtype=np.asarray(Z).dtype)
    out[1:-1, 1:-1] = Z
    return out


# =============================================================================
# Ortigueira fractional centered differences (Section 4.1)
# =============================================================================


def ortigueira_coefficients(order: float, max_offset: int) -> Array:
    """Ortigueira centered coefficients ``g_j^(order)`` for ``j = 0..max_offset``.

    Implements the closed form of Definition "Ortigueira fractional centered
    differences" (Section 4.1),

        g_j^(s) = (-1)^j Gamma(s+1)
                  / [ Gamma(s/2 - j + 1) Gamma(s/2 + j + 1) ].

    The weights are symmetric, ``g_{-j} = g_j``, which is the algebraic root of
    every self-adjointness property in Section 4.4; only ``j >= 0`` is returned.
    At even integer orders the Gamma poles in the denominator make the
    coefficients vanish beyond the classical stencil (e.g. ``s = 2`` gives
    ``g = (2, -1, 0, ...)``, the standard three-point second difference).
    """
    if order <= 0:
        raise ValueError("The Riesz order must be positive.")
    coeffs = np.zeros(max_offset + 1, dtype=float)
    half = order / 2.0
    coeffs[0] = gamma(order + 1.0) / (gamma(half + 1.0) ** 2)
    # Stable recurrence obtained by taking the ratio g_j/g_{j-1}.  Evaluating
    # the two Gamma factors independently overflows for offsets near 170 even
    # though their ratio, and hence the coefficient, is still representable.
    for j in range(1, max_offset + 1):
        coeffs[j] = -((half - j + 1.0) / (half + j)) * coeffs[j - 1]
    return coeffs


# =============================================================================
# Discrete Riesz matrices (Sections 4.1 and 4.7)
# =============================================================================

@lru_cache(maxsize=256)
def _riesz_1d_cached(n: int, h: float, order_rounded: float) -> sparse.csr_matrix:
    """Cached assembly behind :func:`riesz_1d` (see the comments below)."""
    # Assemble the symmetric Toeplitz matrix (g_{p-q}^(s)) and scale by -h^{-s}
    # to obtain the negative-semidefinite discrete Riesz operator of Section 4.1.
    # The cache key rounds the order so repeated calls with the same decimal
    # value reuse the same matrix.
    order = order_rounded
    c = ortigueira_coefficients(order, n - 1)
    T = toeplitz(c)                      # symmetric Toeplitz, first column = c
    D = -(T / (h ** order))              # sign convention: D approximates D^s <= 0
    return sparse.csr_matrix(D)


def riesz_1d(n: int, h: float, order: float) -> sparse.csr_matrix:
    """One-dimensional Ortigueira Riesz matrix on ``n`` interior nodes.

    Returns the symmetric matrix realising ``delta_z^(order)`` in one direction
    (Section 4.1). It is dense in principle but Toeplitz; :class:`Riesz2DCache`
    combines two such matrices via Kronecker products without forming the full
    2-D operator.
    """
    return _riesz_1d_cached(n, float(h), round(float(order), 14))


def clear_riesz_1d_cache() -> None:
    """Clear the process-wide one-dimensional operator cache.

    This is used only by cold-start performance measurements; ordinary solver
    runs benefit from reuse of identical Toeplitz factors.
    """
    _riesz_1d_cached.cache_clear()


class Riesz2DCache:
    """Lazily-built, cached 2-D Riesz operators on a fixed grid.

    Builds and memoises the Kronecker-product operators of Section 4.7,

        D_x^(s) = I_{N_y} (x) D_x1d,   D_y^(s) = D_y1d (x) I_{N_x},

    together with the mixed operator ``D_xy^(sigma) = D_x^(sigma/2) D_y^(sigma/2)``
    (Definition of the discrete mixed Riesz operator). The ``apply_*`` helpers
    apply an operator to an interior field and return an interior field.
    """

    def __init__(self, grid: FDGrid):
        """Create empty operator caches for ``grid``."""
        self.grid = grid
        self._dx: Dict[float, sparse.csr_matrix] = {}
        self._dy: Dict[float, sparse.csr_matrix] = {}
        self._opx: Dict[float, sparse.csr_matrix] = {}
        self._opy: Dict[float, sparse.csr_matrix] = {}
        self._dxy: Dict[float, sparse.csr_matrix] = {}

    def dx1d(self, order: float) -> sparse.csr_matrix:
        """One-dimensional Riesz matrix of the given order in x (cached)."""
        key = round(float(order), 14)
        if key not in self._dx:
            self._dx[key] = riesz_1d(self.grid.nx, self.grid.hx, key)
        return self._dx[key]

    def dy1d(self, order: float) -> sparse.csr_matrix:
        """One-dimensional Riesz matrix of the given order in y (cached)."""
        key = round(float(order), 14)
        if key not in self._dy:
            self._dy[key] = riesz_1d(self.grid.ny, self.grid.hy, key)
        return self._dy[key]

    def Dx(self, order: float) -> sparse.csr_matrix:
        """Two-dimensional Riesz operator in x, ``I_y (x) D_x1d`` (cached)."""
        key = round(float(order), 14)
        if key not in self._opx:
            Iy = sparse.eye(self.grid.ny, format="csr")
            self._opx[key] = sparse.kron(Iy, self.dx1d(key), format="csr")
        return self._opx[key]

    def Dy(self, order: float) -> sparse.csr_matrix:
        """Two-dimensional Riesz operator in y, ``D_y1d (x) I_x`` (cached)."""
        key = round(float(order), 14)
        if key not in self._opy:
            Ix = sparse.eye(self.grid.nx, format="csr")
            self._opy[key] = sparse.kron(self.dy1d(key), Ix, format="csr")
        return self._opy[key]

    def Dxy(self, order: float) -> sparse.csr_matrix:
        """Mixed operator delta_x^(order/2) delta_y^(order/2)."""
        key = round(float(order), 14)
        if key not in self._dxy:
            half = key / 2.0
            self._dxy[key] = (self.Dx(half) @ self.Dy(half)).tocsr()
        return self._dxy[key]

    def apply_x(self, order: float, Z: Array) -> Array:
        """Apply ``D_x^(order)`` to an interior field and return an interior field."""
        return mat(self.Dx(order) @ vec(Z), self.grid)

    def apply_y(self, order: float, Z: Array) -> Array:
        """Apply ``D_y^(order)`` to an interior field and return an interior field."""
        return mat(self.Dy(order) @ vec(Z), self.grid)

    def apply_xy(self, order: float, Z: Array) -> Array:
        """Apply the mixed operator ``D_xy^(order)`` to an interior field."""
        return mat(self.Dxy(order) @ vec(Z), self.grid)


# =============================================================================
# Discrete inner product and norms (Section 4.4)
# =============================================================================

def inner_h(A: Array, B: Array, grid: FDGrid) -> complex:
    """Discrete inner product ``<A,B>_h = hx hy sum_{m,n} A conj(B)``."""
    return grid.hx * grid.hy * np.sum(np.asarray(A) * np.conj(np.asarray(B)))


def l2_norm_h(A: Array, grid: FDGrid) -> float:
    """Discrete ``L^2`` norm induced by :func:`inner_h`."""
    return float(np.sqrt(np.real(inner_h(A, A, grid))))


def l4_power_h(A: Array, grid: FDGrid) -> float:
    """Discrete ``L^4`` fourth power ``hx hy sum |A|^4`` (the cubic energy term)."""
    return float(grid.hx * grid.hy * np.sum(np.abs(A) ** 4))


# =============================================================================
# Classical centered first differences -- momentum diagnostics only (Section 4.12)
#
# These are the integer-order operators ``nabla_x, nabla_y``, kept deliberately
# separate from the fractional ``delta_x^(s), delta_y^(s)`` above: the two
# families do not coincide even at s = 1, and it is precisely their commutator
# with the fractional operators that produces the momentum residuals R_disp/R_lw.
# =============================================================================

def centered_first_x(U: Array, grid: FDGrid) -> Array:
    """Centered first difference in x on the zero-extended field (``nabla_x``)."""
    F = full_with_zero_boundary(U, grid)
    return (F[2:, 1:-1] - F[:-2, 1:-1]) / (2.0 * grid.hx)


def centered_first_y(U: Array, grid: FDGrid) -> Array:
    """Centered first difference in y on the zero-extended field (``nabla_y``)."""
    F = full_with_zero_boundary(U, grid)
    return (F[1:-1, 2:] - F[1:-1, :-2]) / (2.0 * grid.hy)
