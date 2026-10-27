"""Tests for the parameterized Galerkin adapter tiers and factory."""

import pytest

from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)

from typing import Any, Tuple

from numpy.typing import NDArray

from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.compose import compose_galerkin_system
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import LinearAdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import (
    GalerkinPhysicsToODEResidualAdapter,
    skfem_discretization,
)
from pyapprox.pde.models.galerkin import (
    GalerkinPhysicsToODEResidualWithParamJacobianAdapter,
    GalerkinPhysicsToODEResidualWithSetParamAdapter,
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives
from pyapprox.util.backends.numpy import NumpyBkd
from pyapprox.util.linalg.scipy_stack import ScipySparseLU

_Np = NDArray[Any]
_Physics = LinearAdvectionDiffusionReaction[_Np]


def _make_physics(bkd: NumpyBkd) -> _Physics:
    mesh = StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
    basis = LagrangeBasis(mesh, degree=1)
    return LinearAdvectionDiffusionReaction(basis=basis, diffusivity=0.01, bkd=bkd)


class _EvalOnlyParameterization:
    """One parameter, written nowhere, with no derivatives."""

    def __init__(self, physics: _Physics) -> None:
        self._physics = physics

    def nparams(self) -> int:
        return 1

    def physics(self) -> _Physics:
        return self._physics

    def targets(self) -> Tuple[object, ...]:
        return (self._physics,)

    def apply(self, params_1d: _Np) -> None:
        pass

    def owned_coefficients(self) -> Tuple[str, ...]:
        return ("mock_coef",)

    def param_derivatives(self) -> ParamDerivatives[_Np]:
        return ParamDerivatives.none()


class _FirstOrderParameterization(_EvalOnlyParameterization):
    """As the evaluation-only one, with zero first derivatives."""

    def __init__(self, physics: _Physics, bkd: NumpyBkd) -> None:
        super().__init__(physics)
        self._bkd = bkd

    def _jacobian(self, state: _Np, time: float, params_1d: _Np) -> _Np:
        return self._bkd.zeros((self._physics.nstates(), 1))

    def _initial_jacobian(self, params_1d: _Np) -> _Np:
        return self._bkd.zeros((self._physics.nstates(), 1))

    def param_derivatives(self) -> ParamDerivatives[_Np]:
        return ParamDerivatives.first_order(self._jacobian, self._initial_jacobian)


class TestGalerkinAdapterFactoryTiers:
    """Factory selects the fixed adapter tier from the bundle."""

    def test_base_tier_without_parameterization(self, numpy_bkd: NumpyBkd) -> None:
        physics = _make_physics(numpy_bkd)
        adapter = create_galerkin_physics_ode_residual(
            skfem_discretization(compose_galerkin_system(physics)), ScipySparseLU()
        )
        assert type(adapter) is GalerkinPhysicsToODEResidualAdapter
        assert not hasattr(adapter, "param_jacobian")
        assert not hasattr(adapter, "nparams")

    def test_set_param_tier_for_eval_only(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        physics = _make_physics(bkd)
        adapter = create_galerkin_physics_ode_residual(
            skfem_discretization(compose_galerkin_system(physics)),
            ScipySparseLU(),
            _EvalOnlyParameterization(physics),
        )
        assert type(adapter) is GalerkinPhysicsToODEResidualWithSetParamAdapter
        assert not hasattr(adapter, "param_jacobian")
        adapter.set_param(bkd.array([0.5]))
        assert adapter.nparams() == 1

    def test_param_jacobian_tier_for_first_order(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        physics = _make_physics(bkd)
        nstates = physics.nstates()
        adapter = create_galerkin_physics_ode_residual(
            skfem_discretization(compose_galerkin_system(physics)),
            ScipySparseLU(),
            _FirstOrderParameterization(physics, bkd),
        )
        assert isinstance(adapter, GalerkinPhysicsToODEResidualWithParamJacobianAdapter)
        adapter.set_param(bkd.array([0.5]))
        assert adapter.param_jacobian(bkd.zeros((nstates,)), 0.0).shape == (
            nstates,
            1,
        )

    def test_set_param_rejects_2d(self, numpy_bkd: NumpyBkd) -> None:
        bkd = numpy_bkd
        physics = _make_physics(bkd)
        adapter = create_galerkin_physics_ode_residual(
            skfem_discretization(compose_galerkin_system(physics)),
            ScipySparseLU(),
            _EvalOnlyParameterization(physics),
        )
        with pytest.raises(ValueError, match="must be 1D"):
            adapter.set_param(bkd.array([[0.5]]))
