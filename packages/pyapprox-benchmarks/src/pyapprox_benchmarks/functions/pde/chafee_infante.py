"""Chafee-Infante physics builder.

The Chafee-Infante equation is a diffusion equation with cubic
bistable reaction,

    du/dt = gamma * u_xx + lambda * u - u**3,

discretized with Galerkin FEM via ``AdvectionDiffusionReaction``
(zero velocity, reaction ``R(u) = lambda*u - u**3``).

Boundary-condition escalation ladder for the operator-inference
recovery benchmarks (each variant stresses OpInf differently):

(i)   HOMOGENEOUS (this builder's default): zero Dirichlet at the left
      boundary, natural (zero-flux Neumann) at the right.  No input,
      no lift -- the reduced system is autonomous cubic-polynomial in
      the reduced state.
(ii)  Lumped-mass input (follow-on): a time-dependent Dirichlet input
      g(t) at the left with lumped mass kills the g-dot channel, so
      the reduced model gains input monomials but no input-derivative
      column.
(iii) Consistent-mass input (follow-on): consistent mass adds the
      -V^T M l g-dot channel plus joint (z, u) cross monomials from
      the boundary element.

The boundary configuration is a CONSTRUCTOR ARGUMENT, never baked in:
the recovered reduced operator is boundary-condition-specific (the
cubic interacting with a Dirichlet-pinned boundary differs from a
Neumann or periodic one), so revisiting the paper's exact setup is a
one-argument change.
"""

from typing import Any, Callable, List, Optional

import numpy as np
from numpy.typing import NDArray
from pyapprox.pde.constitutive.coefficient_functions import CallableReaction
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.galerkin.protocols.basis import GalerkinBasisProtocol
from pyapprox.pde.galerkin.protocols.boundary import (
    BoundaryConditionProtocol,
)
from pyapprox.util.backends.protocols import Array, Backend


def build_line_basis(
    nx: int,
    bounds: tuple[float, float],
    bkd: Backend[Array],
    degree: int = 1,
) -> LagrangeBasis[Array]:
    """Build a Lagrange basis on the standard (non-periodic) line.

    Parameters
    ----------
    nx : int
        Number of elements (``nx + 1`` P1 dofs).
    bounds : tuple[float, float]
        Domain bounds ``(xmin, xmax)``.
    bkd : Backend[Array]
        Computational backend.
    degree : int, optional
        Polynomial degree of the Lagrange basis. Default 1.

    Returns
    -------
    LagrangeBasis
        Basis on the structured line mesh with named "left"/"right"
        boundaries.
    """
    mesh = StructuredMesh1D(nx=nx, bounds=bounds, bkd=bkd)
    return LagrangeBasis(mesh, degree=degree)


class _CubicBistableReaction:
    """Reaction kernel ``R(x, u) = lam*u - u**3`` (numpy, skfem seam)."""

    def __init__(self, bifurcation: float) -> None:
        self._bifurcation = bifurcation

    def __call__(
        self,
        x: NDArray[np.floating[Any]],
        u: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        result: NDArray[np.floating[Any]] = self._bifurcation * u - u**3
        return result


class _CubicBistableReactionDeriv:
    """Derivative kernel ``R'(x, u) = lam - 3*u**2`` (numpy, skfem seam)."""

    def __init__(self, bifurcation: float) -> None:
        self._bifurcation = bifurcation

    def __call__(
        self,
        x: NDArray[np.floating[Any]],
        u: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        result: NDArray[np.floating[Any]] = self._bifurcation - 3.0 * u**2
        return result


def build_chafee_infante_physics(
    basis: GalerkinBasisProtocol[Array],
    diffusivity: float,
    bifurcation: float,
    bkd: Backend[Array],
    input_func: Optional[Callable[[float], float]] = None,
    bc_kind: str = "dirichlet_neumann",
    forcing: Optional[Callable[..., Any]] = None,
) -> AdvectionDiffusionReaction[Array]:
    """Build Chafee-Infante physics on a shared basis.

    The basis is taken as an argument so parameterized rebuilds share
    ONE discretization (see :func:`build_periodic_burgers_physics`).

    Parameters
    ----------
    basis : GalerkinBasisProtocol[Array]
        Shared finite element basis, typically from
        :func:`build_line_basis`.
    diffusivity : float
        Diffusion coefficient gamma > 0.
    bifurcation : float
        Bifurcation parameter lambda in ``R(u) = lambda*u - u**3``.
        Keep lambda <= ~5 for well-conditioned operator-recovery
        probes; larger values degrade cond(P) (paper Sec. 4.2
        behavior, not a bug).
    bkd : Backend[Array]
        Computational backend.
    input_func : Callable[[float], float], optional
        Dirichlet input ``g(t)`` at the left boundary.  ``None``
        (default) gives the HOMOGENEOUS variant ``g = 0``.
    bc_kind : str, optional
        Boundary configuration.  Only ``"dirichlet_neumann"``
        (Dirichlet at the left, natural Neumann at the right) is
        currently supported; the argument exists so the recovered
        reduced operator's boundary dependence is explicit and other
        configurations are a one-argument extension.
    forcing : Callable, optional
        Forcing term ``f(x)`` or ``f(x, t)`` returning ``(npts,)``
        (numpy at the skfem assembly seam).

    Returns
    -------
    AdvectionDiffusionReaction
        Physics with reaction ``lambda*u - u**3`` and the requested
        boundary configuration.
    """
    if bc_kind != "dirichlet_neumann":
        raise ValueError(
            f"Unsupported bc_kind '{bc_kind}'; only 'dirichlet_neumann' "
            "is implemented (Dirichlet left, natural Neumann right)."
        )

    dirichlet_value: Callable[..., Any] | float
    if input_func is None:
        dirichlet_value = 0.0
    else:
        captured_input = input_func

        def dirichlet_value_func(
            x: NDArray[np.floating[Any]], time: float = 0.0
        ) -> NDArray[np.floating[Any]]:
            return np.full(x.shape[1], float(captured_input(time)))

        dirichlet_value = dirichlet_value_func

    # Natural Neumann at the right needs NO boundary object: the
    # zero-flux term vanishes from the weak form.
    boundary_conditions: List[BoundaryConditionProtocol[Array]] = [
        DirichletBC(basis, "left", dirichlet_value, bkd)
    ]

    return AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=diffusivity,
        bkd=bkd,
        reaction=CallableReaction(
            _CubicBistableReaction(bifurcation),
            _CubicBistableReactionDeriv(bifurcation),
        ),
        forcing=forcing,
        boundary_conditions=boundary_conditions,
    )
