"""Tests for the parameterized Galerkin adapter tiers and factory."""

import pytest
from pyapprox.util.optional_deps import package_available

if not package_available("skfem"):
    pytest.skip("skfem not installed", allow_module_level=True)


from pyapprox.pde.galerkin.basis import LagrangeBasis
from pyapprox.pde.galerkin.mesh import StructuredMesh1D
from pyapprox.pde.galerkin.physics import LinearAdvectionDiffusionReaction
from pyapprox.pde.galerkin.time_integration import (
    GalerkinPhysicsToODEResidualAdapter,
)
from pyapprox.pde.models.galerkin import (
    GalerkinPhysicsToODEResidualWithParamJacobianAdapter,
    GalerkinPhysicsToODEResidualWithSetParamAdapter,
    create_galerkin_physics_ode_residual,
)
from pyapprox.pde.parameterizations.derivatives import ParamDerivatives


class TestGalerkinAdapterFactoryTiers:
    """Factory selects the fixed adapter tier from the bundle."""

    def _make_physics(self, bkd):
        mesh = StructuredMesh1D(nx=8, bounds=(0.0, 1.0), bkd=bkd)
        basis = LagrangeBasis(mesh, degree=1)
        return LinearAdvectionDiffusionReaction(
            basis=basis, diffusivity=0.01, bkd=bkd
        )

    def test_base_tier_without_parameterization(self, numpy_bkd):
        physics = self._make_physics(numpy_bkd)
        adapter = create_galerkin_physics_ode_residual(physics)
        assert type(adapter) is GalerkinPhysicsToODEResidualAdapter
        assert not hasattr(adapter, "param_jacobian")
        assert not hasattr(adapter, "nparams")

    def test_set_param_tier_for_eval_only(self, numpy_bkd):
        bkd = numpy_bkd
        physics = self._make_physics(bkd)

        class EvalOnlyParameterization:
            def __init__(self, physics):
                self._physics = physics

            def nparams(self):
                return 1

            def physics(self):
                return self._physics

            def apply(self, params_1d):
                pass

            def param_derivatives(self):
                return ParamDerivatives.none()

        adapter = create_galerkin_physics_ode_residual(
            physics, EvalOnlyParameterization(physics)
        )
        assert type(adapter) is GalerkinPhysicsToODEResidualWithSetParamAdapter
        assert not hasattr(adapter, "param_jacobian")
        adapter.set_param(bkd.array([0.5]))
        assert adapter.nparams() == 1

    def test_param_jacobian_tier_for_first_order(self, numpy_bkd):
        bkd = numpy_bkd
        physics = self._make_physics(bkd)
        nstates = physics.nstates()

        def _jac(state, time, params_1d):
            return bkd.zeros((nstates, 1))

        def _init_jac(params_1d):
            return bkd.zeros((nstates, 1))

        class FirstOrderParameterization:
            def __init__(self, physics):
                self._physics = physics

            def nparams(self):
                return 1

            def physics(self):
                return self._physics

            def apply(self, params_1d):
                pass

            def param_derivatives(self):
                return ParamDerivatives.first_order(_jac, _init_jac)

        adapter = create_galerkin_physics_ode_residual(
            physics, FirstOrderParameterization(physics)
        )
        assert isinstance(
            adapter, GalerkinPhysicsToODEResidualWithParamJacobianAdapter
        )
        adapter.set_param(bkd.array([0.5]))
        assert adapter.param_jacobian(bkd.zeros((nstates,))).shape == (
            nstates, 1,
        )

    def test_set_param_rejects_2d(self, numpy_bkd):
        bkd = numpy_bkd
        physics = self._make_physics(bkd)

        class EvalOnlyParameterization:
            def __init__(self, physics):
                self._physics = physics

            def nparams(self):
                return 1

            def physics(self):
                return self._physics

            def apply(self, params_1d):
                pass

            def param_derivatives(self):
                return ParamDerivatives.none()

        adapter = create_galerkin_physics_ode_residual(
            physics, EvalOnlyParameterization(physics)
        )
        with pytest.raises(ValueError, match="must be 1D"):
            adapter.set_param(bkd.array([[0.5]]))
