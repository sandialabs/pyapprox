r"""Collocation observations: point values and subdomain integrals.

Setups:

- Old against new: 15 Chebyshev nodes on :math:`[0, 2]`, a random state
  (seed 1). Points 0, 0.7, 1.3 and 2 against ``PointEvaluationFunctional``;
  the integral of :math:`c u` over :math:`[0.3, 1.4]` and over the whole
  domain, :math:`c` random in :math:`[0.5, 1.5]`, against
  ``SubdomainIntegralFunctional``. Values and state Jacobians to 1e-12.
- Derivation-free: at a node the observation is the nodal value
  (1e-14); :math:`x^{14}` is interpolated exactly at 0.7 and 1.3, and
  :math:`\int_{0.3}^{1.4} x^{6} dx` is exact (1e-12); the transpose dot
  test :math:`\langle H u, r \rangle = \langle u, H^\top r \rangle`
  (1e-12).
- One observation, two terms: 1D ADR on :math:`[-1, 1]` (16 nodes,
  two-mode KLE diffusion about 2, velocity 0.3, reaction -0.4, forcing
  :math:`\sin \pi x`, Dirichlet 0.5 and -0.3) solved on the new steady
  path by Newton (atol 1e-12); the forward tangent of
  ``ObservedValue(H)`` at :math:`x = -0.3, 0.45` matches central
  differences of the solved values (minimum relative error below 1e-6),
  and ``EndpointTerm(H)`` reads the last stored state to the same value.
"""

import math
from typing import Callable, Generic, Tuple

import numpy as np
import pytest

from pyapprox.implicitfunction.observation_terms import (
    ObservedValue,
    RowWeights,
    StateObservationProtocol,
)
from pyapprox.implicitfunction.setups import RecomputeAlways
from pyapprox.implicitfunction.state_functional import StateFunctional
from pyapprox.implicitfunction.steady_derivatives import SteadyDerivatives
from pyapprox.implicitfunction.steady_system import SteadyPoint, SteadySystem
from pyapprox.interface.functions.derivative_checks.derivative_checker import (
    DerivativeChecker,
)
from pyapprox.interface.functions.fromcallable.jacobian import (
    FunctionWithJacobianFromCallable,
)
from pyapprox.ode.observation_terms import EndpointTerm
from pyapprox.ode.residual_derivatives import backend_state_space
from pyapprox.pde.collocation.basis import ChebyshevBasis1D
from pyapprox.pde.collocation.boundary.essential_rows import (
    DirichletPart,
    DirichletRows,
)
from pyapprox.pde.collocation.functionals.observations import (
    point_observation,
    subdomain_integral_observation,
)
from pyapprox.pde.collocation.functionals.point_evaluation import (
    PointEvaluationFunctional,
)
from pyapprox.pde.collocation.functionals.subdomain_integral import (
    SubdomainIntegralFunctional,
)
from pyapprox.pde.collocation.mesh import AffineTransform1D, TransformedMesh1D
from pyapprox.pde.collocation.physics.advection_diffusion import (
    AdvectionDiffusionReaction,
)
from pyapprox.pde.collocation.physics.coefficients import NodalField
from pyapprox.pde.field_maps.mesh_kle_field_map import MeshKLEFieldMap
from pyapprox.pde.models.collocation.closed_system import CollocationSystem
from pyapprox.pde.models.collocation.steady_residual import (
    collocation_steady_residual,
)
from pyapprox.pde.parameterizations.collocation_advection_diffusion import (
    CollocationAdvectionDiffusionParameterization,
)
from pyapprox.pde.parameterizations.parameter_state import ParameterState
from pyapprox.util.backends.protocols import Array, Backend
from pyapprox.util.linalg.backend_dense import BackendDenseLU
from pyapprox.util.rootfinding.nonlinear import Newton

_NPTS = 15
_POINTS = [0.0, 0.7, 1.3, 2.0]
_BOUNDS = (0.3, 1.4)


def _basis(bkd: Backend[Array]) -> ChebyshevBasis1D[Array]:
    """15 Chebyshev nodes on [0, 2]."""
    mesh = TransformedMesh1D(_NPTS, bkd, AffineTransform1D((0.0, 2.0), bkd))
    return ChebyshevBasis1D(mesh, bkd)


def _nodes(basis: ChebyshevBasis1D[Array]) -> Array:
    """The physical nodes, shape (npts,)."""
    return basis.mesh().points()[0]


def _random_state(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.random.default_rng(1).normal(0.0, 1.0, (_NPTS, 1)))


def _coefficient(bkd: Backend[Array]) -> Array:
    return bkd.asarray(np.random.default_rng(2).uniform(0.5, 1.5, _NPTS))


def _value(observation: StateObservationProtocol[Array, Array], state: Array) -> Array:
    return ObservedValue(observation).value(state)


def _state_jacobian(bkd: Backend[Array], observation: RowWeights[Array]) -> Array:
    """H as a dense (nobs, npts) matrix, read through its transpose."""
    eye = bkd.eye(observation.nobs())
    return bkd.hstack(
        [
            observation.apply_transpose(eye[:, k : k + 1])
            for k in range(observation.nobs())
        ]
    ).T


class TestAgainstTheOldFunctionals:
    def test_point_values(self, bkd: Backend[Array]) -> None:
        basis, state = _basis(bkd), _random_state(bkd)
        observation = point_observation(basis, _POINTS, bkd)
        param = bkd.zeros((3, 1))
        old = [PointEvaluationFunctional(basis, x, 3, bkd) for x in _POINTS]
        expected = bkd.vstack([functional(state, param) for functional in old])
        bkd.assert_allclose(_value(observation, state), expected, rtol=1e-12)
        bkd.assert_allclose(
            _state_jacobian(bkd, observation),
            bkd.vstack([functional.state_jacobian(state, param) for functional in old]),
            rtol=1e-12,
            atol=1e-14,
        )

    @pytest.mark.parametrize("bounds", [_BOUNDS, None])
    def test_subdomain_integrals(
        self, bkd: Backend[Array], bounds: Tuple[float, float] | None
    ) -> None:
        basis, state, coefficient = _basis(bkd), _random_state(bkd), _coefficient(bkd)
        observation = subdomain_integral_observation(
            basis, bkd, coefficient=coefficient, bounds=bounds
        )
        lower, upper = bounds if bounds is not None else (None, None)
        old = SubdomainIntegralFunctional(
            basis, 3, bkd, a_sub=lower, b_sub=upper, coefficient=coefficient
        )
        param = bkd.zeros((3, 1))
        bkd.assert_allclose(_value(observation, state), old(state, param), rtol=1e-12)
        bkd.assert_allclose(
            _state_jacobian(bkd, observation),
            old.state_jacobian(state, param),
            rtol=1e-12,
            atol=1e-14,
        )


class TestDerivationFree:
    def test_a_node_reads_its_value(self, bkd: Backend[Array]) -> None:
        basis, state = _basis(bkd), _random_state(bkd)
        nodes = _nodes(basis)
        node = bkd.to_float(nodes[4])
        bkd.assert_allclose(
            _value(point_observation(basis, [node], bkd), state),
            state[4:5],
            rtol=1e-14,
            atol=1e-14,
        )

    def test_polynomials_are_exact(self, bkd: Backend[Array]) -> None:
        basis = _basis(bkd)
        nodes = _nodes(basis)
        state = bkd.reshape(nodes**14, (_NPTS, 1))
        # Relative to the nodal data (2**14 at x = 2): 0.7**14 is far
        # smaller, so a pointwise relative bound would test rounding.
        bkd.assert_allclose(
            _value(point_observation(basis, [0.7, 1.3], bkd), state),
            bkd.asarray([[0.7**14], [1.3**14]]),
            rtol=1e-12,
            atol=1e-12 * 2.0**14,
        )
        lower, upper = _BOUNDS
        bkd.assert_allclose(
            _value(
                subdomain_integral_observation(basis, bkd, bounds=_BOUNDS),
                bkd.reshape(nodes**6, (_NPTS, 1)),
            ),
            bkd.asarray([[(upper**7 - lower**7) / 7.0]]),
            rtol=1e-12,
        )

    def test_transpose(self, bkd: Backend[Array]) -> None:
        basis, state = _basis(bkd), _random_state(bkd)
        observation = point_observation(basis, _POINTS, bkd)
        weights = bkd.asarray(np.random.default_rng(3).normal(0.0, 1.0, (4, 1)))
        bkd.assert_allclose(
            bkd.sum(observation.apply(state) * weights),
            bkd.sum(state * observation.apply_transpose(weights)),
            rtol=1e-12,
        )


class _SteadyADR(Generic[Array]):
    """1D ADR with Dirichlet rows and a two-mode KLE diffusion, on the new
    steady path."""

    def __init__(self, bkd: Backend[Array]) -> None:
        self.bkd = bkd
        mesh = TransformedMesh1D(16, bkd)
        self.basis = ChebyshevBasis1D(mesh, bkd)
        physics = AdvectionDiffusionReaction(
            self.basis,
            bkd,
            diffusion=2.0,
            velocity=[bkd.full((16,), 0.3)],
            reaction=-0.4,
            forcing=NodalField(bkd.sin(math.pi * self.basis.nodes())),
        )
        left, right = mesh.boundary_indices(0), mesh.boundary_indices(1)
        essential = DirichletRows(
            bkd, 16, 1, [DirichletPart(left, 0.5), DirichletPart(right, -0.3)]
        )
        rng = np.random.default_rng(5)
        parameterization = CollocationAdvectionDiffusionParameterization(
            physics,
            diffusion_map=MeshKLEFieldMap(
                bkd, bkd.full((16,), 2.0), bkd.asarray(rng.uniform(-0.3, 0.3, (16, 2)))
            ),
            bkd=bkd,
        )
        self.parameters = ParameterState(parameterization)
        self.residual = collocation_steady_residual(
            CollocationSystem(physics, essential), self.parameters
        )
        self.space = backend_state_space(bkd, 16)
        self.system = SteadySystem(
            self.residual, self.space, RecomputeAlways(BackendDenseLU(bkd))
        )
        self.param = bkd.asarray([[0.4], [-0.2]])

    def solve(self, p: Array) -> SteadyPoint[Array, Array]:
        self.parameters.set_param(p[:, 0])
        point = self.system.solve(
            Newton(atol=1e-12, rtol=0.0, maxiters=30), self.bkd.zeros((16, 1))
        )
        assert point.converged()
        return point


def _min_error(
    bkd: Backend[Array],
    nqoi: int,
    sample: Array,
    fun: Callable[[Array], Array],
    jacobian: Callable[[Array], Array],
) -> float:
    errors = DerivativeChecker(
        FunctionWithJacobianFromCallable(
            nqoi=nqoi, nvars=sample.shape[0], fun=fun, jacobian=jacobian, bkd=bkd
        )
    ).check_derivatives(sample, relative=True, central=True)[0]
    return bkd.to_float(bkd.min(errors))


class TestOneObservationTwoTerms:
    def test_steady_tangent(self, bkd: Backend[Array]) -> None:
        case = _SteadyADR(bkd)
        observation = point_observation(case.basis, [-0.3, 0.45], bkd)
        functional: StateFunctional[Array, Array] = StateFunctional.from_terms(
            bkd, 2, state_terms=(ObservedValue(observation),)
        )
        derivs = SteadyDerivatives(case.residual, case.space, functional)

        def value(p: Array) -> Array:
            return derivs.value(case.solve(p))

        def jacobian(p: Array) -> Array:
            point = case.solve(p)
            eye = bkd.eye(2)
            return bkd.hstack(
                [derivs.tangent(point, eye[:, j : j + 1]) for j in range(2)]
            )

        assert _min_error(bkd, 2, case.param, value, jacobian) < 1e-6

    def test_endpoint_reads_the_same_value(self, bkd: Backend[Array]) -> None:
        case = _SteadyADR(bkd)
        observation = point_observation(case.basis, [-0.3, 0.45], bkd)
        state = case.space.copy(case.solve(case.param).solution())
        endpoint = EndpointTerm(observation)
        assert endpoint.reads(7, True) and not endpoint.reads(7, False)
        bkd.assert_allclose(
            endpoint.at(7, 1.0).value(state),
            ObservedValue(observation).value(state),
            rtol=1e-15,
        )


class TestRefusals:
    def test_points(self, bkd: Backend[Array]) -> None:
        basis = _basis(bkd)
        with pytest.raises(ValueError, match="at least one point"):
            point_observation(basis, [], bkd)
        with pytest.raises(ValueError, match="outside the domain"):
            point_observation(basis, [0.5, 2.1], bkd)

    def test_subdomain(self, bkd: Backend[Array]) -> None:
        basis = _basis(bkd)
        with pytest.raises(ValueError, match="increasing"):
            subdomain_integral_observation(basis, bkd, bounds=(1.0, 0.5))
        with pytest.raises(ValueError, match="coefficient must have shape"):
            subdomain_integral_observation(
                basis, bkd, coefficient=bkd.ones((_NPTS - 1,))
            )
