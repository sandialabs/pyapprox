"""Convergence of the essential-boundary flux to the exact flux.

Advection-diffusion-reaction on [-1, 1]^2 with a manufactured solution:
Dirichlet on the left edge, Neumann data from the physics' flux law at
the exact solution on the other three. The flux functional with unit
weights on the left edge's DOFs is the residual tested with
phi_left, the sum of the left hat functions. At the exact solution the
residual tested with any function is minus the law's boundary integral
outside the Neumann edges, and phi_left is 1 on the left edge, so the
target is

    Q* = integral_left law(x, u*, grad u*, n) ds,

the physics' flux law on exact inputs, integrated along the edge at high
order. By Galerkin orthogonality the error of Q converges at twice the
energy rate, h^{2p}: measured 2.0 for P1 and 4.0 for P2 on 4 to 32 cells
per side.

Wrong-model control: the same law evaluated on the discrete solution's
trace and gradient (the flux recovered from grad u_h) converges only at
the gradient's rate, h^p, and fails the same criterion.
"""

from typing import Any, Callable, Dict, List, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from skfem import Functional, asm

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import NormalFluxLawProtocol
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.manufactured import (
    GalerkinManufacturedSolution,
    create_adr_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.pde.models.galerkin.essential_boundary_flux import (
    EssentialBoundaryFluxFunctional,
)
from pyapprox.pde.models.galerkin.parameterized_residual import (
    GalerkinParameterizedResidual,
)
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.adjoint_checks import NumpyArray
from tests._helpers.parameterization_stubs import ApplyOnlyParameterization

_SOL = "sin(1.3*x)*cos(0.7*y) + 0.5*x*y"
_DIFF = 0.7
_VEL = ["1.0+1e-16*x", "0.5+1e-16*x"]
_REACT = 1.0
_MESHES = (4, 8, 16)

# u (1, npts) and grad u (1, 2, npts) at flattened quadrature points
_TraceValues = Callable[[Any, Any], Tuple[Any, Any]]


def _functions(bkd: NumpyBkd) -> Dict[str, Callable[..., Any]]:
    functions, _ = create_adr_manufactured_test(
        bounds=[-1.0, 1.0, -1.0, 1.0],
        sol_str=_SOL,
        diff_str=f"{_DIFF}+1e-16*x",
        react_str=f"{_REACT}*u",
        vel_strs=_VEL,
        bkd=bkd,
    )
    return functions


class _EdgeFluxIntegrand:
    """law(x, u, grad u, n) at the facet quadrature points."""

    __name__ = "edge_flux"

    def __init__(self, law: NormalFluxLawProtocol, values: _TraceValues):
        self._law = law
        self._values = values

    def __call__(self, w: Any) -> Any:
        shape = w.x.shape[1:]
        x = np.asarray(w.x).reshape(2, -1)
        normal = np.asarray(w.n).reshape(2, -1)
        u, grad_u = self._values(x, w)
        flux = self._law.normal_flux(x, u, grad_u, normal, 0.0)
        return np.asarray(flux)[0].reshape(shape)


class _ExactTrace:
    def __init__(self, functions: Dict[str, Callable[..., Any]]) -> None:
        self._functions = functions

    def __call__(self, x: Any, w: Any) -> Tuple[Any, Any]:
        u = np.asarray(self._functions["solution"](x))[None, :]
        grad_u = np.asarray(self._functions["gradient"](x)).T[None, :, :]
        return u, grad_u


def _discrete_trace(x: Any, w: Any) -> Tuple[Any, Any]:
    return (
        np.asarray(w.u_h).reshape(1, -1),
        np.asarray(w.u_h.grad).reshape(1, 2, -1),
    )


def _fluxes(bkd: NumpyBkd, nx: int, degree: int) -> Tuple[float, float, float]:
    """(Q from the functional, the flux recovered from grad u_h, Q*)."""
    mesh = StructuredMesh2D(nx=nx, ny=nx, bounds=[[-1.0, 1.0], [-1.0, 1.0]], bkd=bkd)
    basis = LagrangeBasis(mesh, degree=degree)
    functions = _functions(bkd)
    ms = GalerkinManufacturedSolution(basis, functions, bkd)
    physics = AdvectionDiffusionReaction(
        basis=basis,
        diffusivity=_DIFF,
        bkd=bkd,
        velocity=ms.velocity(),
        reaction=_REACT,
        forcing=ms.forcing(),
    )
    law = physics.flux_law()
    system = compose_galerkin_system(
        physics,
        [
            ms.dirichlet("left"),
            ms.neumann("right", law),
            ms.neumann("bottom", law),
            ms.neumann("top", law),
        ],
    )
    result = SteadyStateSolver(system.steady(), tol=1e-12).solve(
        bkd.zeros((system.nstates(),))
    )
    assert result.converged
    dofs = basis.get_dofs("left")
    functional = EssentialBoundaryFluxFunctional(
        GalerkinParameterizedResidual(
            system.spatial_operator(), ApplyOnlyParameterization(physics)
        ),
        system.steady(),
        dofs,
        bkd.ones((1, dofs.shape[0])),
    )
    flux = bkd.to_float(functional(result.solution[:, None], bkd.zeros((1, 1)))[0, 0])

    facet_basis = basis.skfem_basis().boundary("left", intorder=10)
    u_h = facet_basis.interpolate(bkd.to_numpy(result.solution))
    recovered = float(
        asm(
            Functional(_EdgeFluxIntegrand(law, _discrete_trace)),
            facet_basis,
            u_h=u_h,
        )
    )
    target = float(
        asm(
            Functional(_EdgeFluxIntegrand(law, _ExactTrace(functions))),
            facet_basis,
        )
    )
    return flux, recovered, target


def _rates(errors: List[float]) -> NumpyArray:
    errs = np.asarray(errors)
    return np.log2(errs[:-1] / errs[1:])


class TestFluxConvergence:
    @pytest.mark.parametrize("degree", [1, 2])
    def test_converges_at_twice_the_energy_rate(
        self, numpy_bkd: NumpyBkd, degree: int
    ) -> None:
        results = [_fluxes(numpy_bkd, nx, degree) for nx in _MESHES]
        flux_rates = _rates([abs(q - target) for q, _, target in results])
        recovered_rates = _rates([abs(r - target) for _, r, target in results])
        assert np.all(flux_rates > 2 * degree - 0.2), flux_rates
        assert not np.all(recovered_rates > 2 * degree - 0.2), recovered_rates
