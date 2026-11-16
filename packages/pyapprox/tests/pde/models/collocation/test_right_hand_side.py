"""The collocation right-hand side: an interior operator and its bundle.

The bundle is checked by finite differences, and its Jacobian's declared
structure (affine, time-invariant) by the checker's structure checks. A
nonlinear operator falsely declared linear is the wrong-declaration
control: the checker must refuse it.
"""

from typing import Generic, Optional

import numpy as np
import pytest

from pyapprox.ode.residual_derivatives import STATED_ZERO
from pyapprox.ode.residual_derivatives_check import ODEResidualBundleChecker
from pyapprox.ode.state_derivatives import StateDerivatives
from pyapprox.pde.collocation.basis import ChebyshevBasis1D, ChebyshevBasis2D
from pyapprox.pde.collocation.mesh import TransformedMesh1D, TransformedMesh2D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.burgers import BurgersPhysics1D
from pyapprox.pde.collocation.physics.coefficients import (
    CoefficientInput,
    NodalSupplier,
)
from pyapprox.pde.collocation.protocols.physics import (
    CollocationInteriorOperatorProtocol,
    PhysicsProtocol,
)
from pyapprox.pde.constitutive.coefficient_functions import TimeDependent
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.models.collocation.right_hand_side import (
    CollocationRightHandSide,
)
from pyapprox.pde.parameterizations.collocation_advection_diffusion import (
    CollocationAdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.operators import LinearizationPoint

_TIME = 0.3
_NMODES = 3


class _Interior(Generic[Array]):
    """A physics' residual and Jacobian with the declarations a test
    states for it."""

    def __init__(
        self,
        physics: PhysicsProtocol[Array],
        curvature: StateDerivatives[Array],
        time_invariant: bool,
    ) -> None:
        self._physics = physics
        self._curvature = curvature
        self._time_invariant = time_invariant

    def bkd(self) -> Backend[Array]:
        return self._physics.bkd()

    def nstates(self) -> int:
        return self._physics.nstates()

    def interior_residual(self, state: Array, time: float) -> Array:
        return self._physics.residual(state, time)

    def interior_jacobian(self, state: Array, time: float) -> Array:
        return self._physics.jacobian(state, time)

    def interior_state_derivatives(self) -> StateDerivatives[Array]:
        return self._curvature

    def interior_is_time_invariant(self) -> bool:
        return self._time_invariant

    def interior_jacobian_is_time_invariant(self) -> bool:
        return self._time_invariant

    def owns(self, target: object) -> bool:
        return target is self._physics


def _adr(bkd: Backend[Array]) -> AdvectionDiffusionReaction[Array]:
    basis = ChebyshevBasis2D(TransformedMesh2D(5, 5, bkd), bkd)
    velocity = [bkd.full((basis.npts(),), 0.5), bkd.full((basis.npts(),), -0.2)]
    return AdvectionDiffusionReaction(
        basis, bkd, diffusion=1.5, velocity=velocity, reaction=-0.4
    )


def _burgers(bkd: Backend[Array]) -> BurgersPhysics1D[Array]:
    basis = ChebyshevBasis1D(TransformedMesh1D(9, bkd), bkd)
    return BurgersPhysics1D(basis, bkd, viscosity=0.1)


def _state(bkd: Backend[Array], nstates: int, seed: int = 3) -> Array:
    return bkd.asarray(np.random.default_rng(seed).normal(0.0, 0.5, nstates))


def _parameterized_adr(
    bkd: Backend[Array],
) -> tuple[AdvectionDiffusionReaction[Array], ParameterState[Array]]:
    physics = _adr(bkd)
    npts = physics.npts()
    rng = np.random.default_rng(5)
    field_map = MeshKLEFieldMap(
        bkd,
        bkd.full((npts,), 2.0),
        bkd.asarray(rng.uniform(-0.3, 0.3, (npts, _NMODES))),
    )
    parameterization = CollocationAdvectionDiffusionParameterization(
        physics, diffusion_map=field_map, bkd=bkd
    )
    return physics, ParameterState(parameterization)


class TestJacobianClassification:
    def test_linear_time_invariant_is_constant(self, bkd: Backend[Array]) -> None:
        physics = _adr(bkd)
        rhs = CollocationRightHandSide(
            _Interior(physics, StateDerivatives.linear(bkd), True)
        )
        jacobian = rhs.derivatives().jacobian
        assert jacobian is not None
        assert jacobian.affine() and jacobian.time_invariant()
        state = _state(bkd, physics.nstates())
        bkd.assert_allclose(
            rhs(state, _TIME), physics.residual(state, _TIME), rtol=1e-14
        )
        vec = _state(bkd, physics.nstates(), seed=4)
        snapshot = jacobian(state, _TIME, LinearizationPoint(0, ("test",)))
        bkd.assert_allclose(
            snapshot.apply(vec[:, None])[:, 0],
            physics.jacobian(state, _TIME) @ vec,
            rtol=1e-14,
        )
        report = ODEResidualBundleChecker(rhs).check(state, _TIME)
        assert set(report.structure_errors) == {"affine", "time_invariant"}
        assert report.failures() == []

    def test_linear_time_dependent_is_state_free(self, bkd: Backend[Array]) -> None:
        physics = _adr(bkd)
        rhs = CollocationRightHandSide(
            _Interior(physics, StateDerivatives.linear(bkd), False)
        )
        jacobian = rhs.derivatives().jacobian
        assert jacobian is not None
        assert jacobian.affine() and not jacobian.time_invariant()
        report = ODEResidualBundleChecker(rhs).check(
            _state(bkd, physics.nstates()), _TIME
        )
        assert report.failures() == []

    def test_nonlinear_is_general(self, bkd: Backend[Array]) -> None:
        physics = _burgers(bkd)
        rhs = CollocationRightHandSide(
            _Interior(physics, StateDerivatives.none(), True)
        )
        bundle = rhs.derivatives()
        assert bundle.jacobian is not None and not bundle.jacobian.affine()
        assert bundle.state_state is None
        report = ODEResidualBundleChecker(rhs).check(
            _state(bkd, physics.nstates()), _TIME
        )
        assert report.failures() == []

    def test_nonlinear_declared_linear_is_refused(self, bkd: Backend[Array]) -> None:
        """Wrong-declaration control: Burgers declared linear."""
        physics = _burgers(bkd)
        rhs = CollocationRightHandSide(
            _Interior(physics, StateDerivatives.linear(bkd), True)
        )
        report = ODEResidualBundleChecker(rhs).check(
            _state(bkd, physics.nstates()), _TIME
        )
        assert "affine" in report.failures()


class TestParameterizedBundle:
    def test_second_order_with_declared_curvature(self, bkd: Backend[Array]) -> None:
        physics, parameters = _parameterized_adr(bkd)
        rhs = CollocationRightHandSide(
            _Interior(physics, StateDerivatives.linear(bkd), True),
            parameters,
            initial_state=STATED_ZERO,
        )
        bundle = rhs.derivatives()
        assert bundle.param is not None and bundle.state_state is not None
        state = _state(bkd, physics.nstates())
        adj = _state(bkd, physics.nstates(), seed=6)
        param = bkd.asarray(np.random.default_rng(7).normal(0.0, 0.3, _NMODES))
        report = ODEResidualBundleChecker(rhs).check(state, _TIME, adj=adj, param=param)
        assert report.failures() == []

    def test_first_order_without_curvature(self, bkd: Backend[Array]) -> None:
        """No declared curvature: a first-order bundle, not zeros."""
        physics, parameters = _parameterized_adr(bkd)
        rhs = CollocationRightHandSide(
            _Interior(physics, StateDerivatives.none(), True),
            parameters,
            initial_state=STATED_ZERO,
        )
        bundle = rhs.derivatives()
        assert bundle.param is not None and bundle.state_state is None
        param = bkd.asarray(np.random.default_rng(7).normal(0.0, 0.3, _NMODES))
        report = ODEResidualBundleChecker(rhs).check(
            _state(bkd, physics.nstates()), _TIME, param=param
        )
        assert report.failures() == []

    def test_set_param_moves_the_shared_state(self, bkd: Backend[Array]) -> None:
        physics, parameters = _parameterized_adr(bkd)
        rhs = CollocationRightHandSide(
            _Interior(physics, StateDerivatives.linear(bkd), True),
            parameters,
            initial_state=STATED_ZERO,
        )
        state = _state(bkd, physics.nstates())
        rhs.set_param(bkd.zeros((_NMODES,)))
        before = rhs(state, _TIME)
        version = rhs.parameters_version()
        parameters.set_param(bkd.full((_NMODES,), 0.5))
        assert rhs.parameters_version() == version + 1
        assert bkd.to_float(bkd.norm(rhs(state, _TIME) - before)) > 1e-3


def _growing(coords: Array, time: float) -> Array:
    """A positive coefficient varying in space and time."""
    return (1.0 + time) * (2.0 + coords[0] ** 2)


class _UndeclaredGrowth(Generic[Array]):
    """Varies in time but declares itself time-independent: the
    wrong-declaration control for time invariance."""

    def __init__(self, nodes: Array) -> None:
        self._nodes = nodes

    def values(self, time: float) -> Array:
        return _growing(self._nodes, time)

    def is_time_dependent(self) -> bool:
        return False

    def version(self) -> int:
        return 0


def _adr_with(
    bkd: Backend[Array],
    diffusion: Optional[CoefficientInput[Array]] = 1.5,
    forcing: Optional[CoefficientInput[Array]] = None,
) -> AdvectionDiffusionReaction[Array]:
    basis = ChebyshevBasis2D(TransformedMesh2D(5, 5, bkd), bkd)
    velocity = [bkd.full((basis.npts(),), 0.5), bkd.full((basis.npts(),), -0.2)]
    return AdvectionDiffusionReaction(
        basis,
        bkd,
        diffusion=diffusion,
        velocity=velocity,
        reaction=-0.4,
        forcing=forcing,
    )


def _nodes(bkd: Backend[Array]) -> Array:
    return TransformedMesh2D(5, 5, bkd).points()


def _time_dependent(bkd: Backend[Array]) -> NodalSupplier[Array]:
    supplier: TimeDependent[Array] = TimeDependent(_growing)
    return NodalSupplier(supplier, _nodes(bkd))


class TestADRInterior:
    """ADR is its own interior operator; its declarations follow its
    coefficients'."""

    def test_constant_coefficients_give_a_constant_jacobian(
        self, bkd: Backend[Array]
    ) -> None:
        physics = _adr_with(bkd)
        assert isinstance(physics, CollocationInteriorOperatorProtocol)
        assert physics.interior_is_time_invariant()
        rhs = CollocationRightHandSide(physics)
        jacobian = rhs.derivatives().jacobian
        assert jacobian is not None
        assert jacobian.affine() and jacobian.time_invariant()
        report = ODEResidualBundleChecker(rhs).check(
            _state(bkd, physics.nstates()), _TIME
        )
        assert set(report.structure_errors) == {"affine", "time_invariant"}
        assert report.failures() == []

    def test_time_dependent_diffusion_gives_a_state_free_jacobian(
        self, bkd: Backend[Array]
    ) -> None:
        physics = _adr_with(bkd, diffusion=_time_dependent(bkd))
        assert not physics.interior_jacobian_is_time_invariant()
        rhs = CollocationRightHandSide(physics)
        jacobian = rhs.derivatives().jacobian
        assert jacobian is not None
        assert jacobian.affine() and not jacobian.time_invariant()
        report = ODEResidualBundleChecker(rhs).check(
            _state(bkd, physics.nstates()), _TIME
        )
        assert report.failures() == []

    def test_time_dependent_forcing_keeps_the_jacobian_invariant(
        self, bkd: Backend[Array]
    ) -> None:
        physics = _adr_with(bkd, forcing=_time_dependent(bkd))
        assert physics.interior_jacobian_is_time_invariant()
        assert not physics.interior_is_time_invariant()

    def test_undeclared_time_dependence_is_refused(self, bkd: Backend[Array]) -> None:
        """Wrong-declaration control: a diffusivity varying in time that
        declares itself time-independent."""
        physics = _adr_with(bkd, diffusion=_UndeclaredGrowth(_nodes(bkd)))
        rhs = CollocationRightHandSide(physics)
        report = ODEResidualBundleChecker(rhs).check(
            _state(bkd, physics.nstates()), _TIME
        )
        assert "time_invariant" in report.failures()

    def test_parameterized_adr_is_its_own_interior(self, bkd: Backend[Array]) -> None:
        physics, parameters = _parameterized_adr(bkd)
        rhs = CollocationRightHandSide(physics, parameters, initial_state=STATED_ZERO)
        bundle = rhs.derivatives()
        assert bundle.param is not None and bundle.state_state is not None
        state = _state(bkd, physics.nstates())
        adj = _state(bkd, physics.nstates(), seed=6)
        param = bkd.asarray(np.random.default_rng(7).normal(0.0, 0.3, _NMODES))
        report = ODEResidualBundleChecker(rhs).check(state, _TIME, adj=adj, param=param)
        assert report.failures() == []


class TestRefusals:
    def test_interior_must_satisfy_the_protocol(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        with pytest.raises(TypeError, match="CollocationInteriorOperatorProtocol"):
            CollocationRightHandSide(object())  # type: ignore[arg-type]

    def test_parameterization_of_another_physics(
        self, numpy_bkd: Backend[Array]
    ) -> None:
        _, parameters = _parameterized_adr(numpy_bkd)
        other = _adr(numpy_bkd)
        with pytest.raises(ValueError, match="does not hold"):
            CollocationRightHandSide(
                _Interior(other, StateDerivatives.linear(numpy_bkd), True),
                parameters,
            )

    def test_set_param_without_parameters(self, numpy_bkd: Backend[Array]) -> None:
        rhs = CollocationRightHandSide(
            _Interior(_adr(numpy_bkd), StateDerivatives.linear(numpy_bkd), True)
        )
        with pytest.raises(ValueError, match="without parameters"):
            rhs.set_param(numpy_bkd.zeros((1,)))
