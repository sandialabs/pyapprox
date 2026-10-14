"""Convergence of the time-integrated essential-boundary flux.

Advection-diffusion-reaction on [-1, 1]^2 with a manufactured transient
solution: Dirichlet on the left edge, Neumann data from the physics' flux
law at the exact solution on the other three edges, exact initial state.
The functional with unit weights on the left DOFs converges to

    Q* = int_0^T int_left law(x, t, u*, grad u*, n) ds dt,

the flux law on exact inputs, integrated at high order in space (facet
quadrature) and time (Gauss-Legendre).

The two error sources are separated:

- temporal: a solution quadratic in space, which P2 elements represent
  exactly, leaves only the time-stepping error; the rate is the
  stepper's order;
- spatial: a smooth solution on P1 with Crank-Nicolson at a small step
  gives the h^{2p} rate of the steady flux. Wrong-model control: the
  steady flux integrated over time drops M u_t on the boundary rows and
  converges only at first order in h.
"""

from typing import Any, Callable, Dict, List, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np
from skfem import Functional, asm

from pyapprox.ode.config import TimeIntegrationConfig
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary import NormalFluxLawProtocol
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.manufactured import (
    GalerkinManufacturedSolution,
    create_adr_manufactured_test,
)
from pyapprox.pde.galerkin.mesh import StructuredMesh2D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration.galerkin_model import GalerkinModel
from pyapprox.pde.models.galerkin import (
    GalerkinParameterizedResidual,
    TimeIntegratedEssentialBoundaryFluxFunctional,
)
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.adjoint_checks import NumpyArray
from tests._helpers.parameterization_stubs import ApplyOnlyParameterization

_DIFF = 0.7
_VEL = ["1.0+1e-16*x", "0.5+1e-16*x"]
_REACT = 1.0
_FINAL_TIME = 0.4
# Quadratic in space: P2 represents it exactly, leaving only time error.
_QUADRATIC = "(x**2 + 0.5*x*y - y**2 + 0.3*x)*(1 + sin(2*T))"
_SMOOTH = "sin(1.3*x)*cos(0.7*y)*(1 + sin(2*T)) + 0.5*x*y"


def _functions(bkd: NumpyBkd, sol_str: str) -> Dict[str, Callable[..., Any]]:
    functions, _ = create_adr_manufactured_test(
        bounds=[-1.0, 1.0, -1.0, 1.0],
        sol_str=sol_str,
        diff_str=f"{_DIFF}+1e-16*x",
        react_str=f"{_REACT}*u",
        vel_strs=_VEL,
        bkd=bkd,
        time_dependent=True,
    )
    return functions


class _EdgeFluxIntegrand:
    """law(x, u*, grad u*, n, t) at the facet quadrature points."""

    __name__ = "edge_flux"

    def __init__(
        self,
        law: NormalFluxLawProtocol,
        functions: Dict[str, Callable[..., Any]],
        time: float,
    ) -> None:
        self._law = law
        self._functions = functions
        self._time = time

    def __call__(self, w: Any) -> Any:
        shape = w.x.shape[1:]
        x = np.asarray(w.x).reshape(2, -1)
        normal = np.asarray(w.n).reshape(2, -1)
        u = np.asarray(self._functions["solution"](x, self._time))[None, :]
        grad_u = np.asarray(self._functions["gradient"](x, self._time)).T[None, :, :]
        flux = self._law.normal_flux(x, u, grad_u, normal, self._time)
        return np.asarray(flux)[0].reshape(shape)


def _fluxes(
    bkd: NumpyBkd, sol_str: str, nx: int, degree: int, method: str, deltat: float
) -> Tuple[float, float, float]:
    """(the functional's Q, the steady flux integrated by the rule, Q*)."""
    mesh = StructuredMesh2D(nx=nx, ny=nx, bounds=[[-1.0, 1.0], [-1.0, 1.0]], bkd=bkd)
    basis = LagrangeBasis(mesh, degree=degree)
    functions = _functions(bkd, sol_str)
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
    coords = bkd.to_numpy(basis.dof_coordinates())
    init_state = bkd.asarray(ms.solution()(coords, 0.0))
    config: TimeIntegrationConfig[Any] = TimeIntegrationConfig(
        method=method,
        init_time=0.0,
        final_time=_FINAL_TIME,
        deltat=deltat,
        newton_tol=1e-12,
        newton_maxiter=20,
        lumped_mass=False,
        verbosity=0,
    )
    model = GalerkinModel(system, bkd)
    traj, times = model.solve_transient(init_state, config)
    quadrature = model.last_integrator().time_residual().trajectory_quadrature(times)
    residual = GalerkinParameterizedResidual(
        system.spatial_operator(), ApplyOnlyParameterization(physics)
    )
    dofs = basis.get_dofs("left")
    functional = TimeIntegratedEssentialBoundaryFluxFunctional(
        residual, system, dofs, bkd.ones((1, dofs.shape[0]))
    )
    functional.set_time_quadrature(quadrature)
    param = bkd.zeros((1, 1))
    flux = bkd.to_float(functional(traj, param)[0, 0])

    rule = quadrature.step_residual_rule()
    assert rule is not None
    stages = rule.sample(traj)
    steady_integrated = 0.0
    for jj in range(rule.nsamples()):
        rows = residual.residual(
            stages[:, jj], bkd.to_float(rule.sample_times()[jj]), param[:, 0]
        )
        steady_integrated -= bkd.to_float(rule.weights()[jj] * bkd.sum(rows[dofs]))

    facet_basis = basis.skfem_basis().boundary("left", intorder=10)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    target = sum(
        0.5
        * _FINAL_TIME
        * wt
        * float(
            asm(
                Functional(
                    _EdgeFluxIntegrand(law, functions, 0.5 * _FINAL_TIME * (nd + 1))
                ),
                facet_basis,
            )
        )
        for nd, wt in zip(nodes, weights)
    )
    return flux, steady_integrated, target


def _rates(errors: List[float]) -> NumpyArray:
    errs = np.asarray(errors)
    return np.log2(errs[:-1] / errs[1:])


class TestTemporalConvergence:
    """P2 on 4 x 4 cells, dt = 0.04, 0.02, 0.01 over [0, 0.4]. Measured:
    backward Euler 3.4e-2 -> 8.6e-3 (rate 1.00), Crank-Nicolson
    1.9e-4 -> 1.2e-5 (2.00), implicit midpoint 7.9e-6 -> 5.0e-7 (1.99)."""

    @pytest.mark.parametrize(
        "method,order",
        [("backward_euler", 1), ("crank_nicolson", 2), ("implicit_midpoint", 2)],
    )
    def test_converges_at_the_stepper_order(
        self, numpy_bkd: NumpyBkd, method: str, order: int
    ) -> None:
        errors = [
            abs(q - t)
            for q, _, t in (
                _fluxes(numpy_bkd, _QUADRATIC, 4, 2, method, dt)
                for dt in (0.04, 0.02, 0.01)
            )
        ]
        rates = _rates(errors)
        assert np.all(np.abs(rates - order) < 0.1), (errors, rates)


class TestSpatialConvergence:
    """P1 on 4, 8, 16 cells per side, Crank-Nicolson with dt = 0.005.
    Measured: the functional 3.4e-2 -> 2.1e-3 (rate 2.0, h^{2p}); the
    steady flux integrated over time 0.31 -> 0.08 (rate 1.0)."""

    def test_converges_at_twice_the_energy_rate(self, numpy_bkd: NumpyBkd) -> None:
        results = [
            _fluxes(numpy_bkd, _SMOOTH, nx, 1, "crank_nicolson", 0.005)
            for nx in (4, 8, 16)
        ]
        flux_rates = _rates([abs(q - t) for q, _, t in results])
        steady_rates = _rates([abs(s - t) for _, s, t in results])
        assert np.all(flux_rates > 1.8), flux_rates
        # Wrong-model control: dropping M u_t on the boundary rows.
        assert np.all(steady_rates < 1.2), steady_rates
