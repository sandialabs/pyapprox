"""Stokes solve + velocity extraction for the advection-diffusion problem.

Provides:

- Picklable BC callable adapters :class:`_ParabolicInlet` and
  :class:`_ZeroVelocity` used by :func:`_solve_stokes`. (Replaces a
  pair of local closures that broke ``pickle.dumps`` on the owning
  benchmark.)
- :func:`_solve_stokes` — Navier-Stokes steady solve on an obstructed
  mesh with parabolic inlet + no-slip walls/obstacles.
- :func:`_extract_velocity_callable` — pulls the velocity field from
  a Stokes solution vector and returns a callable that evaluates it
  at arbitrary points (preserving skfem's ``(ndim, nquad, nelem)``
  quadrature-point convention).

No KLE, no OED state, no inference plumbing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
    from pyapprox.pde.galerkin.basis.vector_lagrange import (
        VectorLagrangeBasis,
    )
    from pyapprox.pde.galerkin.mesh.obstructed import ObstructedMesh2D
    from pyapprox.pde.galerkin.physics.stokes import StokesPhysics

import numpy as np
from pyapprox.util.backends.protocols import Array, Backend


class _ParabolicInlet:
    """Inlet velocity profile ``y^(a-1) * (1-y)^(b-1)`` in x-direction.

    Picklable replacement for an equivalent local closure. Callable
    signature matches skfem BC expectations:
    ``(x: np.ndarray, time: float = 0.0) -> np.ndarray``.
    """

    def __init__(self, a: float, b: float) -> None:
        self._a = float(a)
        self._b = float(b)

    def __call__(
        self, x: np.ndarray, time: float = 0.0,
    ) -> np.ndarray:
        y = x[1]
        vals = np.zeros((x.shape[1], 2))
        vals[:, 0] = y ** (self._a - 1) * (1 - y) ** (self._b - 1)
        return vals


class _ZeroVelocity:
    """Zero-velocity BC. Picklable replacement for an equivalent closure."""

    def __call__(
        self, x: np.ndarray, time: float = 0.0,
    ) -> np.ndarray:
        return np.zeros((x.shape[1], 2))


def _solve_stokes(
    mesh: "ObstructedMesh2D[Array]",
    bkd: Backend[Array],
    reynolds_num: float,
    vel_shape_params: List[float],
) -> Tuple[
    Array,
    "StokesPhysics[Array]",
    "VectorLagrangeBasis[Array]",
    "LagrangeBasis[Array]",
]:
    """Solve Stokes on the obstructed mesh for velocity field."""
    from pyapprox.pde.galerkin.basis.lagrange import LagrangeBasis
    from pyapprox.pde.galerkin.basis.vector_lagrange import (
        VectorLagrangeBasis,
    )
    from pyapprox.pde.galerkin.physics.stokes import StokesPhysics
    from pyapprox.pde.galerkin.time_integration.galerkin_model import (
        GalerkinModel,
    )

    vel_basis = VectorLagrangeBasis(mesh, degree=2)
    pres_basis = LagrangeBasis(mesh, degree=1)

    a, b = vel_shape_params
    inlet_func = _ParabolicInlet(a, b)
    zero_vel = _ZeroVelocity()

    # No-slip on obs, bottom, top; parabolic inlet on left.
    vel_bcs: List[Tuple[str, Callable[..., Any]]] = [
        ("left", inlet_func),
        ("bottom", zero_vel),
        ("top", zero_vel),
        ("obs0", zero_vel),
        ("obs1", zero_vel),
        ("obs2", zero_vel),
    ]

    viscosity = 1.0 / reynolds_num

    stokes = StokesPhysics(
        vel_basis,
        pres_basis,
        bkd,
        navier_stokes=True,
        viscosity=viscosity,
        vel_dirichlet_bcs=vel_bcs,
    )

    # NOTE: StokesPhysics does not structurally satisfy
    # GalerkinPhysicsProtocol because it has two bases (velocity and
    # pressure) rather than a single ``basis()``. A proper fix needs a
    # dedicated protocol for vector/multi-field Galerkin physics that
    # both AdvectionDiffusionReaction (single basis) and StokesPhysics
    # (multiple bases) can satisfy. Flagged by mypy; left for a
    # future upstream refactor so the CI ratchet keeps it visible.
    model = GalerkinModel(stokes, bkd)
    init_guess = stokes.init_guess(0.0)
    sol = model.solve_steady(init_guess, tol=1e-10, maxiter=50)

    return sol, stokes, vel_basis, pres_basis


def _extract_velocity_callable(
    sol: Array,
    stokes: "StokesPhysics[Array]",
    vel_basis: "VectorLagrangeBasis[Array]",
    pres_basis: "LagrangeBasis[Array]",
    adr_basis: "LagrangeBasis[Array]",
    bkd: Backend[Array],
    probes_cache: Optional[Dict[int, Any]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Extract velocity from Stokes solution as a callable on the ADR mesh.

    Returns a callable that accepts points of shape ``(2, ...)`` and
    returns velocity of shape ``(2, ...)``, preserving any extra
    dimensions (e.g., quadrature-point structure from skfem:
    ``(2, nqpts, nelem)``).

    Parameters
    ----------
    probes_cache
        Shared ``{npts: probes_csr}`` dict. If provided, the returned
        callable stores and retrieves the skfem ``probes`` sparse
        matrix here instead of in a private dict. Pass the same cache
        to the forcing interpolator so the expensive element-finder is
        run at most once per distinct point count.
    """
    vel_ndofs = stokes.vel_ndofs()
    vel_state_np = bkd.to_numpy(sol[:vel_ndofs])

    # skfem ElementVector DOFs are interleaved:
    # ``[comp0_dof0, comp1_dof0, comp0_dof1, comp1_dof1, ...]``
    vel_x = vel_state_np[0::2]
    vel_y = vel_state_np[1::2]

    vel_skfem = vel_basis.skfem_basis()
    from skfem import Basis

    scalar_elem = vel_skfem.elem.elem
    scalar_vel_basis = Basis(
        vel_basis.mesh().skfem_mesh(),
        scalar_elem,
        intorder=4,
    )

    # Interpolate Stokes velocity onto ADR DOF locations (done once).
    adr_skfem = adr_basis.skfem_basis()
    vel_x_at_dofs = scalar_vel_basis.interpolator(vel_x)(adr_skfem.doflocs)
    vel_y_at_dofs = scalar_vel_basis.interpolator(vel_y)(adr_skfem.doflocs)

    # Probes-matrix cache: keyed by number of evaluation points.
    # The matrix depends only on the mesh and the evaluation points
    # (quadrature coords), which are identical across all time steps
    # and Newton iterations for a given basis.
    cache: Dict[int, Any] = probes_cache if probes_cache is not None else {}

    def velocity_field(x: np.ndarray) -> np.ndarray:
        orig_shape = x.shape[1:]
        x_flat = x.reshape(2, -1)
        npts = x_flat.shape[1]

        if npts not in cache:
            cache[npts] = adr_skfem.probes(x_flat).tocsr()

        probes_csr = cache[npts]
        vx = (probes_csr @ vel_x_at_dofs).reshape(orig_shape)
        vy = (probes_csr @ vel_y_at_dofs).reshape(orig_shape)
        return np.stack([vx, vy], axis=0)

    return velocity_field
