"""Flux balance identities for the scalar Galerkin physics in 1D.

At the solution every non-essential residual row vanishes, so the total
essential-boundary flux (unit weights on both ends) is minus the sum of
all residual rows. Testing with the interpolant of 1 makes every
divergence term telescope to zero, so that sum is the source plus the
non-divergence terms, each computed here independently of the assembly:

    Q_total = -( integral f + integral(non-divergence terms of u) ).

Setup throughout: [0, 1], P1 on 20 cells, f = 1, u(0) = 0, u(1) = 1 (the
nonzero end value makes the advection and Burgers terms nonzero).
Integrals of the P1 solution use the trapezoid rule on the nodes, exact
for P1. Helmholtz and Burgers have no parameterization yet, so they use
an apply-only stub; the balance needs values only.
"""

from typing import Any, Tuple

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

import numpy as np

from pyapprox.pde.constitutive.coefficient_functions import (
    NodalFieldDiffusion,
    TimeIndependent,
)
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.field_maps.transformed import (
    TransformedFieldMap,
    _ExpTransform,
)
from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.boundary.implementations import DirichletBC
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import AdvectionDiffusionReaction
from pyapprox.pde.galerkin.physics.burgers import BurgersPhysics
from pyapprox.pde.galerkin.physics.helmholtz import Helmholtz
from pyapprox.pde.galerkin.physics.quasilinear_diffusion import (
    QuasilinearDiffusion,
)
from pyapprox.pde.galerkin.solvers import SteadyStateSolver
from pyapprox.pde.galerkin.system import GalerkinSystem
from pyapprox.pde.models.galerkin.essential_boundary_flux import (
    EssentialBoundaryFluxFunctional,
)
from pyapprox.pde.models.galerkin.parameterized_residual import (
    GalerkinParameterizedResidual,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.pde.parameterizations.galerkin_advection_diffusion import (
    AdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.galerkin_quasilinear_diffusion import (
    create_quasilinear_diffusivity_parameterization,
)
from pyapprox.util.backends.numpy import NumpyBkd
from tests._helpers.adjoint_checks import NumpyArray

_NX = 20
_PARAM = np.array([0.4, -0.3, 0.2])


def _unit_forcing(x: Any) -> Any:
    return np.ones(x.shape[1])


def _kappa(u: Any) -> Any:
    return 1.0 + u**2


def _kappa_deriv(u: Any) -> Any:
    return 2.0 * u


def _kappa_deriv2(u: Any) -> Any:
    return np.full_like(u, 2.0)


class _ApplyOnly:
    """A parameterization of a physics that has none yet: it targets the
    physics and writes nothing."""

    def __init__(self, physics: Any) -> None:
        self._physics = physics

    def nparams(self) -> int:
        return 1

    def targets(self) -> Tuple[object, ...]:
        return (self._physics,)

    def owned_coefficients(self) -> Tuple[str, ...]:
        return ("none",)

    def apply(self, params_1d: Any) -> None:
        pass

    def param_derivatives(self) -> ParamDerivatives[Any]:
        return ParamDerivatives.none()


def _basis(bkd: NumpyBkd) -> LagrangeBasis[NumpyArray]:
    mesh = StructuredMesh1D(nx=_NX, bounds=(0.0, 1.0), bkd=bkd)
    return LagrangeBasis(mesh, degree=1)


def _log_kle_map(bkd: NumpyBkd, basis: LagrangeBasis[NumpyArray]) -> Any:
    """A positive field exp(W p): three smooth modes on the DOF nodes."""
    coords = bkd.to_numpy(basis.dof_coordinates())[0]
    modes = np.stack(
        [0.4 * np.sin((k + 1) * np.pi * coords) / (k + 1) for k in range(3)],
        axis=1,
    )
    kle = MeshKLEFieldMap(
        bkd, bkd.asarray(np.zeros(coords.shape[0])), bkd.asarray(modes)
    )
    exp = _ExpTransform(bkd)
    return TransformedFieldMap(kle, exp, exp, bkd, transform_deriv2=exp)


def _system(
    bkd: NumpyBkd, basis: LagrangeBasis[NumpyArray], physics: Any
) -> GalerkinSystem[NumpyArray]:
    return compose_galerkin_system(
        physics,
        [
            DirichletBC(basis, "left", 0.0, bkd),
            DirichletBC(basis, "right", 1.0, bkd),
        ],
    )


def _total_flux_and_solution(
    bkd: NumpyBkd,
    basis: LagrangeBasis[NumpyArray],
    system: GalerkinSystem[NumpyArray],
    param_obj: Any,
    params: NumpyArray,
) -> Tuple[float, NumpyArray]:
    """Solve, then evaluate the flux with unit weights on both ends."""
    param_obj.apply(params)
    result = SteadyStateSolver(system.steady(), tol=1e-12).solve(
        bkd.zeros((system.nstates(),))
    )
    assert result.converged
    state = result.solution[:, None]
    ends = bkd.asarray(
        np.array(
            [
                int(bkd.to_numpy(basis.get_dofs("left"))[0]),
                int(bkd.to_numpy(basis.get_dofs("right"))[0]),
            ]
        ),
        dtype=bkd.int64_dtype(),
    )
    flux = EssentialBoundaryFluxFunctional(
        GalerkinParameterizedResidual(system.spatial_operator(), param_obj),
        system.steady(),
        ends,
        bkd.ones((1, 2)),
    )
    return bkd.to_float(flux(state, params[:, None])[0, 0]), state


def _integral(
    bkd: NumpyBkd, basis: LagrangeBasis[NumpyArray], state: NumpyArray
) -> float:
    """integral of the P1 solution over [0, 1] (trapezoid, exact)."""
    x = bkd.to_numpy(basis.dof_coordinates())[0]
    order = np.argsort(x)
    return float(np.trapezoid(bkd.to_numpy(state)[order, 0], x[order]))


class TestScalarFluxBalance:
    def test_advection_diffusion_reaction(self, numpy_bkd: NumpyBkd) -> None:
        """-(kappa u')' + v u' = R(u) + f with R(u) = -2u: the advection
        integral is v (u(1) - u(0)) = v and the reaction integral is
        -2 integral u, so Q = -(1 - 2 integral u - v)."""
        bkd = numpy_bkd
        basis = _basis(bkd)
        velocity = 0.7
        physics = AdvectionDiffusionReaction(
            basis=basis,
            diffusivity=NodalFieldDiffusion(basis),
            bkd=bkd,
            velocity=bkd.asarray([velocity]),
            reaction=-2.0,
            forcing=TimeIndependent(_unit_forcing),
        )
        system = _system(bkd, basis, physics)
        param_obj = AdvectionDiffusionParameterization(
            physics, diffusivity_map=_log_kle_map(bkd, basis), bkd=bkd
        )
        flux, state = _total_flux_and_solution(
            bkd, basis, system, param_obj, bkd.asarray(_PARAM)
        )
        expected = -(1.0 - 2.0 * _integral(bkd, basis, state) - velocity)
        bkd.assert_allclose(bkd.asarray([flux]), bkd.asarray([expected]), rtol=1e-10)

    def test_helmholtz(self, numpy_bkd: NumpyBkd) -> None:
        """-u'' + k^2 u = f: the k^2 u term gives Q = -(1 - k^2 integral
        u)."""
        bkd = numpy_bkd
        basis = _basis(bkd)
        wavenumber = 3.0
        physics = Helmholtz(
            basis, wavenumber, bkd, forcing=TimeIndependent(_unit_forcing)
        )
        system = _system(bkd, basis, physics)
        flux, state = _total_flux_and_solution(
            bkd, basis, system, _ApplyOnly(physics), bkd.zeros((1,))
        )
        expected = -(1.0 - wavenumber**2 * _integral(bkd, basis, state))
        bkd.assert_allclose(bkd.asarray([flux]), bkd.asarray([expected]), rtol=1e-10)

    def test_burgers(self, numpy_bkd: NumpyBkd) -> None:
        """u u' = nu u'' + f: the integral of u u' is [u^2/2] = 1/2, so
        Q = -(1 - 1/2)."""
        bkd = numpy_bkd
        basis = _basis(bkd)
        physics = BurgersPhysics(
            basis, 0.5, bkd, forcing=TimeIndependent(_unit_forcing)
        )
        system = _system(bkd, basis, physics)
        flux, _ = _total_flux_and_solution(
            bkd, basis, system, _ApplyOnly(physics), bkd.zeros((1,))
        )
        bkd.assert_allclose(bkd.asarray([flux]), bkd.asarray([-0.5]), rtol=1e-10)

    def test_quasilinear_diffusion(self, numpy_bkd: NumpyBkd) -> None:
        """-(a(x) kappa(u) u')' = f is pure divergence form: Q = -1,
        whatever a(x) and kappa(u)."""
        bkd = numpy_bkd
        basis = _basis(bkd)
        physics = QuasilinearDiffusion(
            basis,
            NodalFieldDiffusion(basis),
            bkd,
            _kappa,
            _kappa_deriv,
            _kappa_deriv2,
            forcing=TimeIndependent(_unit_forcing),
        )
        system = _system(bkd, basis, physics)
        param_obj = create_quasilinear_diffusivity_parameterization(
            physics, _log_kle_map(bkd, basis), bkd
        )
        flux, _ = _total_flux_and_solution(
            bkd, basis, system, param_obj, bkd.asarray(_PARAM)
        )
        bkd.assert_allclose(bkd.asarray([flux]), bkd.asarray([-1.0]), rtol=1e-10)
